import datetime
import subprocess

Timer=datetime.time(hour=9,minute=0,second=0)
current=datetime.datetime.now().time()

if current>=Timer:
    print("Its Time")
    news=subprocess.run(["python",'-u',"/home/masterfool1386/Desktop/Projects/CyberDaily/scheduler.py"],capture_output=True,text=True)
    print(news.stdout)
else:
    print("Not Now")