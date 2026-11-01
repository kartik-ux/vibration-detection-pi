import json, os, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from .config import CLASSES, Settings
from .features import extract_features, FEATURE_NAMES
from .simulator import VibrationSimulator


def collect(settings: Settings, output=None, windows_per_class=None):
    n = windows_per_class or settings.windows_per_class
    sim=VibrationSimulator(settings.sample_rate_hz, settings.random_seed)
    records=[]
    for signal,label in sim.dataset(n, settings.samples_per_window):
        row=extract_features(signal, settings.sample_rate_hz); row["label"]=label; records.append(row)
    df=pd.DataFrame(records); path=output or settings.data_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True); df.to_csv(path,index=False)
    return df,path


def train(settings: Settings, data_path=None, model_path=None):
    data_path=data_path or settings.data_path; model_path=model_path or settings.model_path
    df=pd.read_csv(data_path)
    X=df[FEATURE_NAMES]; y=df.label
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,random_state=settings.random_seed,stratify=y)
    clf=Pipeline([("scale",StandardScaler()),("classifier",RandomForestClassifier(n_estimators=240,max_features="sqrt",min_samples_leaf=1,class_weight="balanced",random_state=settings.random_seed,n_jobs=-1))])
    clf.fit(X_train,y_train); pred=clf.predict(X_test)
    # Unsupervised model learns healthy operating envelope only.
    anomaly=IsolationForest(n_estimators=180,contamination=0.04,random_state=settings.random_seed)
    anomaly.fit(X_train[y_train=="healthy"])
    report={"accuracy":float(accuracy_score(y_test,pred)),"classes":list(CLASSES),"classification_report":classification_report(y_test,pred,output_dict=True),"confusion_matrix":confusion_matrix(y_test,pred,labels=CLASSES).tolist(),"test_samples":int(len(y_test)),"features":FEATURE_NAMES}
    os.makedirs(os.path.dirname(model_path) or ".",exist_ok=True)
    joblib.dump({"classifier":clf,"anomaly_detector":anomaly,"features":FEATURE_NAMES,"sample_rate_hz":settings.sample_rate_hz,"version":"1.0.0"},model_path)
    report_path=os.path.join(settings.reports_dir,"training_report.json"); os.makedirs(settings.reports_dir,exist_ok=True)
    with open(report_path,"w") as f: json.dump(report,f,indent=2)
    return report,model_path,report_path


def predict_window(signal, model, sample_rate_hz=None):
    start=time.perf_counter()
    sr=sample_rate_hz or model.get("sample_rate_hz",800)
    row=extract_features(signal,sr); X=pd.DataFrame([row],columns=model["features"])
    clf=model["classifier"]; label=str(clf.predict(X)[0])
    probs=clf.predict_proba(X)[0]; confidence=float(max(probs))
    anomaly_score=float(model["anomaly_detector"].decision_function(X)[0])
    is_anomaly=bool(model["anomaly_detector"].predict(X)[0] == -1)
    return {"prediction":label,"confidence":confidence,"anomaly":is_anomaly,"anomaly_score":anomaly_score,"latency_ms":(time.perf_counter()-start)*1000}


def detect(settings: Settings, model_path=None, windows=20, output=None):
    model=joblib.load(model_path or settings.model_path)
    sim=VibrationSimulator(settings.sample_rate_hz,seed=991)
    labels=["healthy","imbalance","misalignment","bearing_fault","looseness"]
    records=[]
    for i in range(windows):
        truth=labels[i%len(labels)]; signal=sim.window(truth,settings.samples_per_window)
        result=predict_window(signal,model,settings.sample_rate_hz)
        result.update({"ground_truth":truth,"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())})
        records.append(result)
    df=pd.DataFrame(records); path=output or os.path.join(settings.reports_dir,"detection_log.csv")
    os.makedirs(os.path.dirname(path) or ".",exist_ok=True); df.to_csv(path,index=False)
    return df,path


def benchmark(settings: Settings, model_path=None, iterations=300):
    model=joblib.load(model_path or settings.model_path); sim=VibrationSimulator(settings.sample_rate_hz,seed=2026)
    samples=[sim.window("healthy" if i%2==0 else "bearing_fault",settings.samples_per_window) for i in range(iterations)]
    timings=[]; predictions=[]
    for signal in samples:
        result=predict_window(signal,model,settings.sample_rate_hz); timings.append(result["latency_ms"]); predictions.append(result["prediction"])
    arr=np.asarray(timings)
    result={"iterations":iterations,"mean_latency_ms":float(arr.mean()),"median_latency_ms":float(np.median(arr)),"p95_latency_ms":float(np.percentile(arr,95)),"max_latency_ms":float(arr.max()),"throughput_windows_per_second":float(1000/arr.mean()),"estimated_model_size_bytes":int(os.path.getsize(model_path or settings.model_path)),"note":"Host-machine benchmark; Raspberry Pi performance must be measured on the target device."}
    os.makedirs(settings.reports_dir,exist_ok=True)
    with open(os.path.join(settings.reports_dir,"benchmark.json"),"w") as f: json.dump(result,f,indent=2)
    return result
