import argparse, json
from .config import Settings
from .pipeline import collect, train, detect, benchmark

def main():
    p=argparse.ArgumentParser(prog="vibration-pi",description="Edge AI vibration monitoring pipeline")
    p.add_argument("command",choices=["collect","train","detect","benchmark","all"])
    p.add_argument("--windows-per-class",type=int,default=100)
    p.add_argument("--windows",type=int,default=25)
    p.add_argument("--iterations",type=int,default=300)
    args=p.parse_args(); s=Settings(windows_per_class=args.windows_per_class)
    if args.command in ("collect","all"):
        df,path=collect(s); print(f"Collected {len(df)} feature windows -> {path}")
    if args.command in ("train","all"):
        report,model,report_path=train(s); print(f"Model saved -> {model}\nTest accuracy: {report['accuracy']:.3f}\nReport -> {report_path}")
    if args.command in ("detect","all"):
        df,path=detect(s,windows=args.windows); print(f"Detection log -> {path}\n{df[['ground_truth','prediction','confidence','anomaly']].to_string(index=False)}")
    if args.command in ("benchmark","all"):
        result=benchmark(s,iterations=args.iterations); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
