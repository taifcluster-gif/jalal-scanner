#!/usr/bin/env python3
# جلال سكانر v3.1
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings, threading, webbrowser, os, json
warnings.filterwarnings("ignore")

app = Flask(__name__)

# ══ إعدادات ══
def load_keys():
    keys = {"API_KEY":"","SECRET_KEY":""}
    kf = os.path.join(os.path.dirname(os.path.abspath(__file__)),"binance_keys")
    if os.path.exists(kf):
        with open(kf) as f:
            for line in f:
                if "=" in line:
                    k,v = line.strip().split("=",1)
                    keys[k.strip()] = v.strip()
    return keys

KEYS = load_keys()
CUSTOM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"custom_stocks.json")

def load_custom():
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return {"tadawul":{},"us":{},"crypto":{},"excluded":[]}

def save_custom(d):
    with open(CUSTOM_FILE,"w",encoding="utf-8") as f:
        json.dump(d,f,ensure_ascii=False,indent=2)

DEFAULT_TADAWUL = {
    "5110":"الكهرباء","7020":"علم","1211":"معادن","4020":"العقارية",
    "7010":"STC","4072":"MBC","2082":"أكوا باور","1080":"الأهلي",
    "3050":"أسمنت الجنوب","4220":"إعمار","2223":"أديس","8030":"إعادة",
    "2140":"مبكو","4030":"البحري","1010":"الرياض","3040":"أسمنت القصيم",
    "3090":"أسمنت الشرقية","2220":"مسار","2150":"مرافق","2222":"أرامكو",
    "4280":"المملكة","4040":"سابتكو","1050":"الإنماء","3060":"أسمنت ينبع",
    "4090":"طيبة","2040":"الخزف","2020":"سابك للمغذيات","2010":"سابك",
    "3030":"أسمنت السعودية","4190":"جرير","2280":"المراعي","4261":"بدجت",
    "2285":"سدافكو","4240":"العثيم","4130":"الحبيب","1120":"الراجحي",
}
DEFAULT_US = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Google","META":"Meta",
    "NVDA":"NVIDIA","AMD":"AMD","TSLA":"Tesla","AMZN":"Amazon",
    "ENPH":"Enphase","FSLR":"First Solar","NEE":"NextEra",
    "JNJ":"Johnson","UNH":"UnitedHealth","ABBV":"AbbVie",
    "COST":"Costco","HD":"Home Depot","JPM":"JPMorgan","V":"Visa",
    "MA":"Mastercard","PG":"Procter Gamble",
}
DEFAULT_CRYPTO = {
    "BTCUSDT":"Bitcoin","ETHUSDT":"Ethereum","BNBUSDT":"BNB",
    "SOLUSDT":"Solana","XRPUSDT":"Ripple","ADAUSDT":"Cardano",
    "DOTUSDT":"Polkadot","LINKUSDT":"Chainlink","AVAXUSDT":"Avalanche",
    "MATICUSDT":"Polygon","ATOMUSDT":"Cosmos","UNIUSDT":"Uniswap",
    "LTCUSDT":"Litecoin","XLMUSDT":"Stellar","ALGOUSDT":"Algorand",
    "VETUSDT":"VeChain","FILUSDT":"Filecoin","AAVEUSDT":"Aave",
    "SANDUSDT":"Sandbox","MANAUSDT":"Decentraland",
}
BENCHMARK = {"tadawul":"^TASI.SR","us":"^GSPC","crypto":"BTC-USD"}

# ══ القطاعات ══
SECTORS = {
    "البنوك":         ["1010","1020","1030","1050","1060","1080","1120","1140","1150","1180"],
    "البتروكيماويات": ["2010","2020","2060","2070","2080","2090","2100","2110","2150","2170","2180","2220","2223"],
    "الطاقة":         ["2222","5110","5120"],
    "الاتصالات":      ["7010","7020","7030","7040"],
    "التجزئة":        ["4190","4200","4210","4220","4230","4240","4260","4270"],
    "الأسمنت":        ["3010","3020","3030","3040","3050","3060","3080","3090"],
    "الصناعة":        ["2140","2160","2200","2210","2280","2285","4030","4040","4072"],
    "العقارات":       ["4020","4090","4100","4110","4130","4150","4160","4220"],
}

def get_sector_heat(data):
    """حساب حرارة القطاعات بناءً على متوسط الـ score"""
    sector_scores = {}
    for sector, codes in SECTORS.items():
        scores = [s["score"] for s in data if s["code"] in codes]
        if scores:
            avg = round(sum(scores)/len(scores), 1)
            buy_count = sum(1 for s in data if s["code"] in codes and s["verdict"]=="BUY")
            sector_scores[sector] = {"avg": avg, "count": len(scores), "buy": buy_count}
    return dict(sorted(sector_scores.items(), key=lambda x: -x[1]["avg"]))


scan_state = {
    "tadawul":{"data":None,"last_scan":None,"status":"idle"},
    "us":{"data":None,"last_scan":None,"status":"idle"},
    "crypto":{"data":None,"last_scan":None,"status":"idle"},
}

# ══ حساب ══
def get_df(t,p,i):
    try:
        df=yf.download(t,period=p,interval=i,progress=False,auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def sma(s,p): return s.rolling(p).mean()

def rsi_f(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def macd_f(s):
    ml=ema(s,12)-ema(s,26); sg=ema(ml,9); return ml,sg,ml-sg

def adx_f(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    at=tr.rolling(p).mean()
    up,dn=h.diff(),-l.diff()
    dp=pd.Series(np.where((up>dn)&(up>0),up,0.),index=h.index).rolling(p).mean()
    dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=h.index).rolling(p).mean()
    dip,dim=100*dp/at,100*dm/at
    return (100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)).rolling(p).mean()

def stoch_f(h,l,c,k=14,d=3):
    kv=100*(c-l.rolling(k).min())/(h.rolling(k).max()-l.rolling(k).min()).replace(0,np.nan)
    return kv,kv.rolling(d).mean()

def atr_f(h,l,c,p=14):
    return pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1).rolling(p).mean()

def dur(trend,atr_v,price,tp1):
    if atr_v<=0: return "—"
    days=max(1,round((tp1-price)/atr_v))
    if trend=="استثمار": return str(days)+"-"+str(days*2)+" يوم"
    elif trend=="سوينج": return str(days)+"-"+str(days+3)+" أيام"
    elif trend=="مضاربة": return str(max(1,days-1))+"-"+str(days+2)+" أيام"
    return "—"

def rs_calc(sdf,bdf,period=20):
    try:
        if sdf.empty or bdf.empty: return 0,"محايد →"
        sr=sdf["Close"].pct_change(period).iloc[-1]
        br=bdf["Close"].pct_change(period).iloc[-1]
        rs=round((sr-br)*100,2)
        if rs>5: lbl="قوي جداً ↑↑"
        elif rs>2: lbl="قوي ↑"
        elif rs>-2: lbl="محايد →"
        elif rs>-5: lbl="ضعيف ↓"
        else: lbl="ضعيف جداً ↓↓"
        return rs,lbl
    except: return 0,"محايد →"

def find_obs(df,lb=50):
    try:
        d=df.tail(lb).copy(); obs=[]
        for i in range(2,len(d)-1):
            if (d["Close"].iloc[i]<d["Open"].iloc[i] and
                d["Close"].iloc[i+1]>d["Open"].iloc[i+1] and
                d["High"].iloc[i+1]>d["High"].iloc[i]):
                obs.append({"high":round(float(d["High"].iloc[i]),3),
                            "low":round(float(d["Low"].iloc[i]),3),
                            "mid":round((float(d["High"].iloc[i])+float(d["Low"].iloc[i]))/2,3)})
        return obs[-2:] if len(obs)>=2 else obs
    except: return []

def find_fvgs(df,lb=30):
    try:
        d=df.tail(lb).copy(); fvgs=[]
        for i in range(1,len(d)-1):
            if float(d["Low"].iloc[i+1])>float(d["High"].iloc[i-1]):
                fvgs.append({"top":round(float(d["Low"].iloc[i+1]),3),
                             "bottom":round(float(d["High"].iloc[i-1]),3),
                             "mid":round((float(d["Low"].iloc[i+1])+float(d["High"].iloc[i-1]))/2,3)})
        return fvgs[-2:] if len(fvgs)>=2 else fvgs
    except: return []

def is_us_session():
    now=datetime.utcnow(); h=now.hour; wd=now.weekday()
    if wd>=5: return False,"السوق مغلق — عطلة"
    if 13<=h<20: return True,"London + NY مفتوح 🟢"
    if 8<=h<13: return True,"London مفتوح 🟡"
    return False,"السوق مغلق 🔴"

def analyze(code,name,market,bdf=None):
    try:
        ticker=code+".SR" if market=="tadawul" else code
        dfd=get_df(ticker,"2y","1d"); dfw=get_df(ticker,"5y","1wk"); dfm=get_df(ticker,"10y","1mo")
        if dfd.empty or len(dfd)<50: return None
        price=float(dfd["Close"].iloc[-1])
        e20d=float(ema(dfd["Close"],20).iloc[-1])
        e20w=float(ema(dfw["Close"],20).iloc[-1]) if len(dfw)>=20 else None
        e20m=float(ema(dfm["Close"],20).iloc[-1]) if len(dfm)>=20 else None
        ad=price>e20d; aw=price>e20w if e20w else False; am=price>e20m if e20m else False
        if ad and aw and am: trend,stars="استثمار",3
        elif ad and aw: trend,stars="سوينج",2
        elif ad: trend,stars="مضاربة",1
        elif not ad and not aw and not am: trend,stars="تجنب",0
        else: trend,stars="انتظر",0
        d=dfd.copy()
        e20=ema(d["Close"],20); e50=ema(d["Close"],50); e200=ema(d["Close"],200)
        rv=rsi_f(d["Close"]); ml,sg,mh=macd_f(d["Close"])
        adv=adx_f(d["High"],d["Low"],d["Close"])
        sk,sdv=stoch_f(d["High"],d["Low"],d["Close"])
        va=sma(d["Volume"],20); bm=sma(d["Close"],20)
        bw=(4*d["Close"].rolling(20).std())/bm
        av=atr_f(d["High"],d["Low"],d["Close"])
        c1=price>float(e20.iloc[-1]); c2=float(e20.iloc[-1])>float(e50.iloc[-1])
        c3=price>float(e200.iloc[-1]); c4=float(ml.iloc[-1])>float(sg.iloc[-1])
        c5=float(mh.iloc[-1])>float(mh.iloc[-2]); c6=40<=float(rv.iloc[-1])<=70
        c7=float(rv.iloc[-1])>float(rv.iloc[-2]); c8=float(adv.iloc[-1])>20
        c9=float(sk.iloc[-1])>20 and float(sk.iloc[-1])>float(sdv.iloc[-1])
        c10=float(d["Volume"].iloc[-1])>float(va.iloc[-1])*1.2
        c11=price>float(bm.iloc[-1]); c12=float(bw.iloc[-1])>float(bw.iloc[-2])
        vr=float(d["Volume"].iloc[-1])/float(va.iloc[-1]) if float(va.iloc[-1])>0 else 1
        liq=round(price*float(va.iloc[-1])/1e6,1)
        liq_lbl=("عالية 🟢 "+str(liq)+"م") if liq>100 else (("متوسطة 🟡 "+str(liq)+"م") if liq>10 else ("منخفضة 🔴 "+str(liq)+"م"))
        score=(2 if c1 else 0)+(2 if c2 else 0)+(2 if c3 else 0)+(2 if c4 else 0)+\
              (1 if c5 else 0)+(2 if c6 else 0)+(1 if c7 else 0)+(2 if c8 else 0)+\
              (2 if c9 else 0)+(2 if c10 else 0)+(1 if c11 else 0)+(1 if c12 else 0)
        atr_v=float(av.iloc[-1]); p=round(price,3)
        lb=round(p*0.995,3)
        t1s=round(p+atr_v*1.8,3); t1l=round(p+atr_v*2.0,3)
        t2s=round(p+atr_v*3.8,3); t2l=round(p+atr_v*4.0,3)
        ss=round(p-atr_v*1.3,3); sl=round(p-atr_v*1.5,3)
        rr=round((t1l-p)/max(p-sl,0.001),2)
        psl=round((p-sl)/p*100,2); ptp=round((t1l-p)/p*100,2)
        tp=round((atr_v*1.5/p)*100,2); tg=round(atr_v*0.3,3)
        be=round(p+(p-sl)*0.1,3)
        if score>=15 and rr>=1.3: v,pr,bt="BUY",1,"🟢 BUY"; note=""
        elif score>=13 and rr>=1.0 and ad and aw:
            v,pr,bt="BUY",1,"🟡 BUY مشروط"
            cr=[]
            if rr<1.3: cr.append("R:R="+str(rr))
            if not am: cr.append("الشهري ضعيف")
            if score<15: cr.append("Score "+str(score))
            note=" | ".join(cr)
        elif score>=10: v,pr,bt,note="WAIT",2,"⏳ WAIT",""
        else: v,pr,bt,note="AVOID",3,"🔴 AVOID",""
        rsv,rsl=rs_calc(dfd,bdf) if bdf is not None and not bdf.empty else (0,"محايد →")
        conds=[
            {"n":"Close > EMA20","ok":c1,"w":2},{"n":"EMA20 > EMA50","ok":c2,"w":2},
            {"n":"Close > EMA200","ok":c3,"w":2},{"n":"MACD > Signal","ok":c4,"w":2},
            {"n":"MACD Rising","ok":c5,"w":1},{"n":"RSI (40-70)","ok":c6,"w":2},
            {"n":"RSI Rising","ok":c7,"w":1},{"n":"ADX > 20","ok":c8,"w":2},
            {"n":"Stoch > 20","ok":c9,"w":2},{"n":"Vol > Avg×1.2","ok":c10,"w":2},
            {"n":"Above BB Mid","ok":c11,"w":1},{"n":"BB Expanding","ok":c12,"w":1},
        ]
        return {
            "code":code,"name":name,"price":p,"market":market,"stars":stars,"trend":trend,
            "above_daily":ad,"above_weekly":aw,"above_monthly":am,
            "score":score,"score_pct":round(score/20*100),"verdict":v,"priority":pr,"bt":bt,"note":note,
            "lb":lb,"t1s":t1s,"t1l":t1l,"t2s":t2s,"t2l":t2l,"ss":ss,"sl":sl,
            "rr":rr,"psl":psl,"ptp":ptp,"tp":tp,"tg":tg,"be":be,
            "dur":dur(trend,atr_v,p,t1l),"rsi":round(float(rv.iloc[-1]),1),"adx":round(float(adv.iloc[-1]),1),
            "vr":round(vr,1),"exp":vr>=3.0,"liq":liq,"liq_lbl":liq_lbl,
            "rsv":rsv,"rsl":rsl,"obs":find_obs(dfd),"fvgs":find_fvgs(dfd),
            "conds":conds,"cok":sum(1 for c in conds if c["ok"]),"is_custom":False,
        }
    except: return None

def get_crypto_px(sym):
    try:
        ak=KEYS.get("API_KEY","")
        if ak:
            import urllib.request,json as j
            url="https://api.binance.com/api/v3/ticker/24hr?symbol="+sym
            req=urllib.request.Request(url,headers={"X-MBX-APIKEY":ak})
            with urllib.request.urlopen(req,timeout=5) as r:
                d=j.loads(r.read())
                return {"price":float(d["lastPrice"]),"chg":float(d["priceChangePercent"])}
    except: pass
    try:
        ys=sym.replace("USDT","-USD"); df=get_df(ys,"5d","1h")
        if not df.empty:
            p=float(df["Close"].iloc[-1]); p2=float(df["Close"].iloc[-24]) if len(df)>=24 else p
            return {"price":p,"chg":round((p-p2)/p2*100,2)}
    except: pass
    return None

def analyze_crypto(sym,name,bdf=None):
    try:
        info=get_crypto_px(sym)
        if not info: return None
        price=info["price"]; chg=info["chg"]
        ys=sym.replace("USDT","-USD"); dfd=get_df(ys,"2y","1d")
        if dfd.empty or len(dfd)<50: return None
        e20d=float(ema(dfd["Close"],20).iloc[-1])
        dfw=get_df(ys,"5y","1wk")
        e20w=float(ema(dfw["Close"],20).iloc[-1]) if len(dfw)>=20 else None
        ad=price>e20d; aw=price>e20w if e20w else False
        if ad and aw: trend,stars="سوينج",2
        elif ad: trend,stars="مضاربة",1
        else: trend,stars="تجنب",0
        d=dfd.copy(); rv=rsi_f(d["Close"]); av=atr_f(d["High"],d["Low"],d["Close"])
        atr_v=float(av.iloc[-1]); p=round(price,4)
        lb=round(p*0.995,4); t1s=round(p+atr_v*1.8,4); t1l=round(p+atr_v*2.0,4)
        t2s=round(p+atr_v*3.8,4); t2l=round(p+atr_v*4.0,4)
        ss=round(p-atr_v*1.3,4); sl=round(p-atr_v*1.5,4)
        rr=round((t1l-p)/max(p-sl,0.001),2)
        psl=round((p-sl)/p*100,2); ptp=round((t1l-p)/p*100,2)
        tp=round((atr_v*1.5/p)*100,2); tg=round(atr_v*0.3,4); be=round(p+(p-sl)*0.1,4)
        score=10 if (ad and aw) else (6 if ad else 3)
        if score>=8 and rr>=1.3: v,pr,bt,note="BUY",1,"🟢 BUY",""
        elif score>=6 and ad: v,pr,bt,note="BUY",1,"🟡 BUY مشروط",""
        elif score>=5: v,pr,bt,note="WAIT",2,"⏳ WAIT",""
        else: v,pr,bt,note="AVOID",3,"🔴 AVOID",""
        rsv,rsl=rs_calc(dfd,bdf) if bdf is not None else (0,"محايد →")
        return {
            "code":sym,"name":name,"price":p,"market":"crypto","stars":stars,"trend":trend,
            "above_daily":ad,"above_weekly":aw,"above_monthly":False,"chg":chg,
            "score":score,"score_pct":round(score/20*100),"verdict":v,"priority":pr,"bt":bt,"note":note,
            "lb":lb,"t1s":t1s,"t1l":t1l,"t2s":t2s,"t2l":t2l,"ss":ss,"sl":sl,
            "rr":rr,"psl":psl,"ptp":ptp,"tp":tp,"tg":tg,"be":be,
            "dur":dur(trend,atr_v,p,t1l),"rsi":round(float(rv.iloc[-1]),1),"adx":0,
            "vr":1,"exp":chg>10,"liq":0,"liq_lbl":"—","rsv":rsv,"rsl":rsl,
            "obs":find_obs(dfd),"fvgs":find_fvgs(dfd),
            "conds":[],"cok":0,"is_custom":False,
        }
    except: return None

def run_scan(market):
    scan_state[market]["status"]="scanning"
    custom=load_custom(); excl=custom.get("excluded",[])
    bdf=get_df(BENCHMARK.get(market,""),"2y","1d")
    results=[]
    if market=="tadawul":
        stocks={**DEFAULT_TADAWUL,**custom.get("tadawul",{})}
        for code,name in stocks.items():
            if code in excl: continue
            r=analyze(code,name,"tadawul",bdf)
            if r: r["is_custom"]=code in custom.get("tadawul",{}); results.append(r)
    elif market=="us":
        stocks={**DEFAULT_US,**custom.get("us",{})}
        for code,name in stocks.items():
            if code in excl: continue
            r=analyze(code,name,"us",bdf)
            if r: r["is_custom"]=code in custom.get("us",{}); results.append(r)
    elif market=="crypto":
        stocks={**DEFAULT_CRYPTO,**custom.get("crypto",{})}
        for code,name in stocks.items():
            if code in excl: continue
            r=analyze_crypto(code,name,bdf)
            if r: r["is_custom"]=code in custom.get("crypto",{}); results.append(r)
    results.sort(key=lambda x:(x["priority"],-x["score"],-x["stars"],-x.get("rsv",0)))
    scan_state[market]["data"]=results[:20]
    scan_state[market]["last_scan"]=datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state[market]["status"]="done"

def cv(v): return '<span class="cv" onclick="copyVal(this)">'+str(v)+'</span>'

def row_html(s,i):
    vl=s["verdict"].lower()
    sfx=".SR" if s["market"]=="tadawul" else ""
    bc="rb-buy" if s["verdict"]=="BUY" and "مشروط" not in s["bt"] else (
       "rb-cond" if "مشروط" in s["bt"] else (
       "rb-wait" if s["verdict"]=="WAIT" else "rb-avoid"))
    dc="pos" if s["above_daily"] else "neg"
    wc="pos" if s["above_weekly"] else "neg"
    mc="pos" if s["above_monthly"] else "neg"
    exp=""
    if s.get("exp"): exp='<span class="exp-dot" title="حجم شاذ">🚀</span>'
    sc_cls="sc-hi" if s["score"]>=15 else ("sc-md" if s["score"]>=10 else "sc-lo")
    stars="⭐"*s["stars"]

    row=(
        '<div class="srow '+vl+'" onclick="toggle('+str(i)+')" id="row'+str(i)+'">'
        '<div class="sr-l">'
        '<span class="'+bc+'">'+s["bt"]+'</span>'
        '<span class="sr-name">'+s["name"]+'</span>'
        '<span class="sr-code">'+s["code"]+sfx+'</span>'
        +exp+
        '</div>'
        '<div class="sr-m"><span class="sr-stars">'+stars+'</span><span class="sr-tr">'+s["trend"]+'</span></div>'
        '<div class="sr-r">'
        '<span class="sr-price">'+str(s["price"])+'</span>'
        '<span class="sr-sc '+sc_cls+'">'+str(s["score"])+'/20</span>'
        '<span class="sr-fr"><span class="'+dc+'">'+("✓" if s["above_daily"] else "✗")+'</span>'
        '<span class="'+wc+'">'+("✓" if s["above_weekly"] else "✗")+'</span>'
        '<span class="'+mc+'">'+("✓" if s["above_monthly"] else "✗")+'</span></span>'
        '<span class="sr-rsi">RSI:'+str(s["rsi"])+'</span>'
        '<span class="sr-chev" id="ch'+str(i)+'">▼</span>'
        '</div>'
        '</div>'
    )

    # Detail card
    note_html=""
    if s.get("note"): note_html='<div class="cond-note">⚠️ التحفظ: '+s["note"]+'</div>'

    rs_cls="pos" if s.get("rsv",0)>2 else ("neg" if s.get("rsv",0)<-2 else "muted")

    # OB
    ob_html=""
    if s.get("obs"):
        ob_html='<div class="ds"><div class="dt">📦 Order Blocks</div>'
        ob_html+='<div class="pro-exp">المنطقة اللي دخل منها المال الكبير — لو السعر رجع إليها وارتد = فرصة دخول 🎯</div>'
        for ob in s["obs"]:
            dist=round(((s["price"]-ob["mid"])/s["price"])*100,1)
            arrow="⬇️" if dist>0 else "⬆️"
            prox="قريب 🔥" if abs(dist)<3 else ("متوسط" if abs(dist)<8 else "بعيد")
            ob_html+='<div class="ob-item"><div><span class="ob-z">'+str(ob["low"])+" — "+str(ob["high"])+'</span>'
            ob_html+='<div class="ob-d">'+arrow+" "+str(abs(dist))+"% | "+prox+'</div></div>'
            ob_html+='<span class="ob-m">وسط: '+cv(ob["mid"])+'</span></div>'
        ob_html+='</div>'

    # FVG
    fvg_html=""
    if s.get("fvgs"):
        fvg_html='<div class="ds"><div class="dt">⚡ Fair Value Gaps</div>'
        fvg_html+='<div class="pro-exp">فجوة تركها السعر بسرعة — غالباً يرجع يملأها قبل ما يكمل الصعود 📍</div>'
        for fvg in s["fvgs"]:
            dist=round(((s["price"]-fvg["mid"])/s["price"])*100,1)
            arrow="⬇️" if dist>0 else "⬆️"
            filled=s["price"]>=fvg["bottom"] and s["price"]<=fvg["top"]
            st="✅ السعر داخلها" if filled else ("قريب 🔥" if abs(dist)<3 else "لم تُملأ بعد")
            fvg_html+='<div class="ob-item"><div><span class="ob-z">'+str(fvg["bottom"])+" — "+str(fvg["top"])+'</span>'
            fvg_html+='<div class="ob-d">'+arrow+" "+str(abs(dist))+"% | "+st+'</div></div>'
            fvg_html+='<span class="ob-m">وسط: '+cv(fvg["mid"])+'</span></div>'
        fvg_html+='</div>'

    # Conditions
    conds_html=""
    for c in s.get("conds",[]):
        cls="co" if c["ok"] else "cf"
        dcl="dok" if c["ok"] else "dfail"
        conds_html+='<div class="ci '+cls+'"><div class="cd '+dcl+'"></div><span>'+c["n"]+'</span><span class="cw">★'+str(c["w"])+'</span></div>'

    del_btn=""
    if s.get("is_custom"):
        del_btn='<button class="del-btn" onclick="event.stopPropagation();delStock(\''+s["code"]+'\',\''+s["market"]+'\')">🗑 حذف</button>'
    excl_btn='<button class="excl-btn" onclick="event.stopPropagation();exclStock(\''+s["code"]+'\')">⊘ استبعاد</button>'

    detail=(
        '<div class="det" id="det'+str(i)+'">'
        +note_html+
        '<div class="dq">'
        '<div class="dqi"><div class="dql">القوة النسبية 📊</div><div class="dqv '+rs_cls+'">'+s.get("rsl","—")+'</div><div class="dql" style="font-size:0.54rem;color:var(--mut);margin-top:1px;">مقارنة بمؤشر السوق</div></div>'
        '<div class="dqi"><div class="dql">السيولة اليومية 💧</div><div class="dqv">'+s.get("liq_lbl","—")+'</div><div class="dql" style="font-size:0.54rem;color:var(--mut);margin-top:1px;">متوسط التداول بالريال</div></div>'
        '<div class="dqi"><div class="dql">ADX — قوة الاتجاه</div><div class="dqv">'+str(s["adx"])+'</div><div class="dql" style="font-size:0.54rem;color:var(--mut);margin-top:1px;">&gt;40 قوي | 20-40 معقول | &lt;20 عرضي</div></div>'
        '<div class="dqi"><div class="dql">المدة المتوقعة ⏱</div><div class="dqv teal">'+s.get("dur","—")+'</div><div class="dql" style="font-size:0.54rem;color:var(--mut);margin-top:1px;">تقدير للوصول لـ TP1</div></div>'
        '</div>'

        '<div class="det-grid">'
        '<div class="det-col">'
        '<div class="ds"><div class="dt">🛒 أمر الشراء</div>'
        '<div class="og1"><div class="oi"><div class="ol">سعر الأمر (Limit)</div>'
        '<div class="ov ev">'+cv(s["lb"])+'</div><div class="oh">اضغط للنسخ</div></div></div></div>'

        '<div class="ds"><div class="dt">💰 جني الربح (Take Profit)</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov tv">'+cv(s["t1s"])+'</div><div class="oh">Trigger</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov tv">'+cv(s["t1l"])+'</div><div class="oh pos">+'+str(s["ptp"])+'%</div></div>'
        '</div>'
        '<div class="og2" style="margin-top:5px;">'
        '<div class="oi"><div class="ol">إيقاف 2</div><div class="ov tv">'+cv(s["t2s"])+'</div></div>'
        '<div class="oi"><div class="ol">أمر 2</div><div class="ov tv">'+cv(s["t2l"])+'</div><div class="oh pos">+'+str(round(s["ptp"]*2,2))+'%</div></div>'
        '</div></div>'

        '<div class="ds ds-sl"><div class="dt">🛡 وقف الخسارة (Stop Loss)</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov sv">'+cv(s["ss"])+'</div><div class="oh">Trigger</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov sv">'+cv(s["sl"])+'</div><div class="oh neg">-'+str(s["psl"])+'%</div></div>'
        '</div>'
        '<div class="rr-row"><span>نسبة المخاطرة/العائد R:R: <strong style="color:var(--gold)">'+str(s["rr"])+'</strong><span style="font-size:0.6rem;color:var(--mut)"> (1.3+ جيد)</span></span><span>⏱ '+s.get("dur","—")+'</span></div>'
        '</div>'
        '</div>'

        '<div class="det-col">'
        '<div class="ds ds-tr"><div class="dt">🚀 Trailing Stop</div>'
        '<div class="og3">'
        '<div class="oi"><div class="ol">نوع التتبع</div><div class="ov" style="color:var(--teal)">نسبة مئوية</div></div>'
        '<div class="oi"><div class="ol">المبلغ / النسبة</div><div class="ov sv">'+cv(str(s["tp"])+"%")+'</div><div class="oh">أدخل هذه النسبة</div></div>'
        '<div class="oi"><div class="ol">الفارق السعري</div><div class="ov" style="color:var(--muted)">'+cv(s["tg"])+'</div></div>'
        '</div>'
        '<div class="tr-note">سعر الدخول: '+str(s["price"])+'  |  الهدف: '+str(s["t2l"])+'</div>'
        '</div>'

        '<div class="ds"><div class="dt">📊 إدارة المركز</div>'
        '<div class="peg">'
        '<div class="pei"><div class="pel">عند TP1 — بيع</div><div class="pev pos">50%</div></div>'
        '<div class="pei"><div class="pel">عند TP2 — بيع</div><div class="pev pos">30%</div></div>'
        '<div class="pei"><div class="pel">اترك تجري</div><div class="pev teal">20%</div></div>'
        '</div>'
        '<div class="be-n">🔒 Break Even: حرك SL إلى '+str(s["be"])+'  بعد وصول TP1</div>'
        '</div>'
        '</div>'
        '</div>'

        +ob_html+fvg_html+

        ('<details onclick="event.stopPropagation()"><summary class="cb">📊 الشروط ('+str(s["cok"])+'/12)</summary>'
         '<div class="cleg">★★=2  |  ★=1</div><div class="cg">'+conds_html+'</div></details>'
         if conds_html else "")

        +'<div class="btn-row">'+del_btn+excl_btn+'</div>'
        '</div>'
    )
    return row+detail

PAGE_TOP = """<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>جلال سكانر v3.1</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0a0e1a;--card:#131d35;--bdr:#1e2d4a;--gold:#f5c518;--green:#00e676;--red:#ff3d57;--yel:#ffb300;--blue:#2979ff;--teal:#00bcd4;--txt:#e8eaf6;--mut:#7986cb;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--txt);font-family:'IBM Plex Sans Arabic',sans-serif;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(41,121,255,0.05),transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(0,188,212,0.05),transparent 60%);pointer-events:none;z-index:0;}
.wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:18px;}
.hdr{text-align:center;padding:20px 20px 14px;border-bottom:1px solid var(--bdr);margin-bottom:14px;}
.hdr h1{font-size:clamp(1.4rem,4vw,2.3rem);font-weight:700;background:linear-gradient(135deg,var(--gold),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:2px;margin-bottom:3px;}
.sub{color:var(--mut);font-size:0.77rem;}.ls{color:var(--mut);font-size:0.7rem;margin-top:2px;}
.sess{text-align:center;padding:5px;font-size:0.74rem;border-radius:7px;margin-bottom:11px;}
.tabs{display:flex;gap:6px;justify-content:center;margin-bottom:14px;flex-wrap:wrap;}
.tab-btn{padding:8px 20px;border-radius:50px;border:1px solid var(--bdr);background:transparent;color:var(--mut);font-family:inherit;font-size:0.86rem;cursor:pointer;transition:all 0.3s;}
.tab-btn:hover{border-color:var(--blue);color:var(--txt);}
.tab-btn.active{background:linear-gradient(135deg,#1a3a6e,#0d2044);border-color:var(--blue);color:var(--txt);}
.tab-c{display:none;}.tab-c.active{display:block;}
.scan-btn{display:inline-flex;align-items:center;gap:6px;background:linear-gradient(135deg,#1a3a6e,#0d2044);border:1px solid var(--blue);color:var(--txt);padding:9px 22px;border-radius:50px;font-size:0.88rem;font-family:inherit;cursor:pointer;margin-top:9px;transition:all 0.3s;}
.scan-btn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(41,121,255,0.3);}
/* Stats chips */
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-bottom:14px;}
.chip{flex:1;min-width:75px;background:var(--card);border:2px solid var(--bdr);border-radius:11px;padding:9px 10px;text-align:center;cursor:pointer;transition:all 0.2s;}
.chip:hover{transform:translateY(-2px);}
.chip.af{transform:translateY(-2px);box-shadow:0 4px 18px rgba(0,0,0,0.35);}
.chip .num{font-size:1.5rem;font-weight:700;}.chip .lbl{font-size:0.63rem;color:var(--mut);margin-top:2px;}
.chip.cg{border-color:var(--green)}.chip.cg .num{color:var(--green)}.chip.cg.af{background:rgba(0,230,118,0.08);}
.chip.cy{border-color:var(--yel)}.chip.cy .num{color:var(--yel)}.chip.cy.af{background:rgba(255,179,0,0.08);}
.chip.cr{border-color:var(--red)}.chip.cr .num{color:var(--red)}.chip.cr.af{background:rgba(255,61,87,0.08);}
.chip.co{border-color:var(--gold)}.chip.co .num{color:var(--gold)}.chip.co.af{background:rgba(245,197,24,0.08);}
/* Explosion */
.exp-sec{background:rgba(245,197,24,0.04);border:1px solid rgba(245,197,24,0.2);border-radius:11px;padding:11px;margin-bottom:13px;}
.exp-ttl{color:var(--gold);font-size:0.88rem;font-weight:700;margin-bottom:8px;}
.exp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:6px;}
.exp-card{background:rgba(0,0,0,0.2);border:1px solid rgba(245,197,24,0.16);border-radius:8px;padding:7px 10px;display:flex;align-items:center;justify-content:space-between;gap:6px;font-size:0.82rem;}
.exp-nm{font-weight:600;}.exp-cd{color:var(--mut);font-size:0.65rem;}
.exp-px{color:var(--gold);font-weight:700;}
.exp-rt{background:rgba(245,197,24,0.1);color:var(--yel);border:1px solid rgba(245,197,24,0.2);padding:2px 6px;border-radius:6px;font-size:0.68rem;font-weight:600;}
/* List */
.slist{display:flex;flex-direction:column;gap:5px;}
.srow{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px 14px;cursor:pointer;transition:all 0.18s;display:flex;align-items:center;justify-content:space-between;gap:10px;}
.srow:hover{border-color:var(--blue);background:#152040;}
.srow.hidden{display:none;}
.srow.buy{border-right:3px solid var(--green);}
.srow.wait{border-right:3px solid var(--yel);}
.srow.avoid{border-right:3px solid var(--red);}
.sr-l{display:flex;align-items:center;gap:7px;min-width:0;flex:1;}
.sr-m{display:flex;align-items:center;gap:5px;flex-shrink:0;}
.sr-r{display:flex;align-items:center;gap:8px;flex-shrink:0;}
.sr-name{font-weight:600;font-size:0.93rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:120px;}
.sr-code{color:var(--mut);font-size:0.65rem;flex-shrink:0;}
.sr-stars{color:var(--gold);font-size:0.76rem;}
.sr-tr{color:var(--mut);font-size:0.68rem;}
.sr-price{color:var(--gold);font-weight:700;font-size:0.92rem;}
.sr-sc{font-weight:700;font-size:0.78rem;padding:2px 6px;border-radius:7px;}
.sc-hi{color:var(--green);background:rgba(0,230,118,0.1);}
.sc-md{color:var(--yel);background:rgba(255,179,0,0.1);}
.sc-lo{color:var(--red);background:rgba(255,61,87,0.1);}
.sr-fr{font-size:0.74rem;display:flex;gap:3px;}
.sr-rsi{color:var(--mut);font-size:0.68rem;}
.sr-chev{color:var(--mut);font-size:0.68rem;transition:transform 0.25s;}
.sr-chev.op{transform:rotate(180deg);}
.rb-buy{background:rgba(0,230,118,0.1);color:var(--green);border:1px solid var(--green);padding:2px 8px;border-radius:9px;font-size:0.68rem;font-weight:600;white-space:nowrap;flex-shrink:0;}
.rb-cond{background:rgba(255,179,0,0.1);color:var(--yel);border:1px solid var(--yel);padding:2px 8px;border-radius:9px;font-size:0.68rem;font-weight:600;white-space:nowrap;flex-shrink:0;}
.rb-wait{background:rgba(255,179,0,0.07);color:var(--yel);border:1px solid rgba(255,179,0,0.3);padding:2px 8px;border-radius:9px;font-size:0.68rem;white-space:nowrap;flex-shrink:0;}
.rb-avoid{background:rgba(255,61,87,0.1);color:var(--red);border:1px solid var(--red);padding:2px 8px;border-radius:9px;font-size:0.68rem;white-space:nowrap;flex-shrink:0;}
.exp-dot{font-size:0.72rem;}
/* Detail */
.det{display:none;background:rgba(0,0,0,0.12);border:1px solid var(--bdr);border-radius:0 0 11px 11px;padding:13px;margin-top:-5px;border-top:none;}
.det.op{display:block;}
.det-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:8px;}
.det-col{display:flex;flex-direction:column;gap:7px;}
.dq{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-bottom:11px;}
.dqi{background:rgba(0,0,0,0.18);border-radius:7px;padding:6px;text-align:center;}
.dql{font-size:0.58rem;color:var(--mut);margin-bottom:2px;}.dqv{font-size:0.8rem;font-weight:600;}
.teal{color:var(--teal);}
.ds{background:rgba(0,0,0,0.18);border:1px solid var(--bdr);border-radius:9px;padding:9px;margin-bottom:7px;}
.ds-sl{border-color:rgba(255,61,87,0.2);background:rgba(255,61,87,0.025);}
.ds-tr{border-color:rgba(0,188,212,0.2);background:rgba(0,188,212,0.025);}
.dt{font-size:0.64rem;color:var(--mut);font-weight:600;margin-bottom:7px;letter-spacing:0.3px;}
.og1{display:grid;grid-template-columns:1fr;gap:5px;}
.og2{display:grid;grid-template-columns:1fr 1fr;gap:5px;}
.og3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;}
.oi{text-align:center;}.ol{font-size:0.58rem;color:var(--mut);margin-bottom:2px;}
.ov{font-size:0.9rem;font-weight:700;padding:2px 4px;border-radius:5px;display:inline-block;}
.ev{color:var(--txt)}.tv{color:var(--green)}.sv{color:var(--red)}
.oh{font-size:0.56rem;color:var(--mut);margin-top:1px;}
.cv{cursor:pointer;border-radius:4px;padding:1px 3px;transition:all 0.12s;}
.cv:hover{background:rgba(255,255,255,0.07);}
.cv:active{transform:scale(0.92);}
.copied{background:rgba(0,230,118,0.14)!important;color:var(--green)!important;}
.rr-row{display:flex;justify-content:space-between;font-size:0.67rem;color:var(--mut);margin-top:6px;padding-top:5px;border-top:1px solid var(--bdr);}
.tr-note{font-size:0.6rem;color:var(--teal);margin-top:5px;padding-top:5px;border-top:1px solid rgba(0,188,212,0.16);}
.pos{color:var(--green)}.neg{color:var(--red)}.muted{color:var(--mut)}
.pro-exp{font-size:0.63rem;color:var(--mut);background:rgba(121,134,203,0.05);padding:4px 6px;border-radius:5px;margin-bottom:6px;border-right:2px solid var(--blue);}
.cond-note{font-size:0.63rem;color:var(--yel);background:rgba(255,179,0,0.05);padding:3px 7px;border-radius:5px;margin-bottom:7px;border-right:2px solid var(--yel);}
.ob-item{display:flex;justify-content:space-between;align-items:flex-start;padding:3px 5px;border-radius:5px;background:rgba(41,121,255,0.04);margin-bottom:3px;font-size:0.67rem;}
.ob-z{color:var(--blue)}.ob-m{color:var(--mut);font-size:0.61rem;}.ob-d{font-size:0.58rem;color:var(--mut);margin-top:1px;}
.peg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:6px;}
.pei{text-align:center;background:rgba(0,0,0,0.16);border-radius:6px;padding:4px;}
.pel{font-size:0.57rem;color:var(--mut)}.pev{font-size:0.88rem;font-weight:700;margin-top:1px;}
.be-n{font-size:0.61rem;color:var(--teal);background:rgba(0,188,212,0.04);padding:4px 7px;border-radius:5px;border:1px solid rgba(0,188,212,0.1);}
.cb{background:none;border:1px solid var(--bdr);color:var(--mut);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.64rem;cursor:pointer;width:100%;margin-top:3px;}
.cleg{font-size:0.57rem;color:var(--mut);margin-top:5px;opacity:0.6;}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-top:5px;}
.ci{display:flex;align-items:center;gap:3px;font-size:0.61rem;padding:2px 4px;border-radius:4px;background:rgba(255,255,255,0.02);}
.co{color:var(--green)}.cf{color:var(--red);opacity:0.55;}
.cd{width:4px;height:4px;border-radius:50%;flex-shrink:0;}
.dok{background:var(--green)}.dfail{background:var(--red);opacity:0.4;}
.cw{color:var(--gold);font-size:0.56rem;margin-right:auto;}
.btn-row{display:flex;gap:5px;margin-top:6px;}
.del-btn{background:rgba(255,61,87,0.07);border:1px solid rgba(255,61,87,0.2);color:var(--red);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.64rem;cursor:pointer;}
.excl-btn{background:rgba(121,134,203,0.07);border:1px solid rgba(121,134,203,0.2);color:var(--mut);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.64rem;cursor:pointer;}
/* Custom panel */
.cp{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px;margin-bottom:13px;}
.cp h3{font-size:0.81rem;color:var(--teal);margin-bottom:8px;}
.ci-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px;}
.ci-inp{background:rgba(0,0,0,0.28);border:1px solid var(--bdr);color:var(--txt);padding:6px 9px;border-radius:7px;font-family:inherit;font-size:0.77rem;flex:1;min-width:80px;}
.ci-inp:focus{outline:none;border-color:var(--blue);}
.add-btn{background:rgba(0,230,118,0.09);border:1px solid var(--green);color:var(--green);padding:6px 11px;border-radius:7px;cursor:pointer;font-family:inherit;font-size:0.77rem;}
.file-row{display:flex;align-items:center;gap:7px;margin-top:5px;}
.file-lbl{background:rgba(41,121,255,0.09);border:1px solid var(--blue);color:var(--blue);padding:5px 11px;border-radius:7px;cursor:pointer;font-size:0.73rem;white-space:nowrap;}
.file-hint{font-size:0.66rem;color:var(--mut);}
.ecl{display:inline-block;background:rgba(255,61,87,0.06);border:1px solid rgba(255,61,87,0.18);color:var(--red);padding:2px 7px;border-radius:11px;font-size:0.64rem;cursor:pointer;margin:2px;}
.ecl:hover{background:rgba(255,61,87,0.15);}
/* Loading */
.lo{position:fixed;inset:0;background:rgba(10,14,26,0.93);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:100;}
.lo.show{display:flex;}
.lr{width:56px;height:56px;border:3px solid var(--bdr);border-top-color:var(--teal);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:11px;}
@keyframes spin{to{transform:rotate(360deg)}}
.lt{color:var(--teal);font-size:0.88rem;}.ls2{color:var(--mut);font-size:0.69rem;margin-top:4px;}
.lgd{background:rgba(0,0,0,0.16);border:1px solid var(--bdr);border-radius:10px;padding:10px;margin-bottom:13px;font-size:0.7rem;color:var(--mut);}
.lgd h4{color:var(--teal);margin-bottom:6px;font-size:0.76rem;}
.lgd-g{display:grid;grid-template-columns:repeat(auto-fill,minmax(185px,1fr));gap:4px;}
.nr{text-align:center;padding:42px 20px;color:var(--mut);}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:var(--green);color:#000;padding:5px 14px;border-radius:14px;font-size:0.74rem;font-weight:600;opacity:0;transition:opacity 0.3s;z-index:200;pointer-events:none;}
.toast.show{opacity:1;}
@media(max-width:768px){
  .wrap{padding:10px;}.sr-m{display:none;}.sr-rsi{display:none;}
  .det-grid{grid-template-columns:1fr;}
  .dq{grid-template-columns:1fr 1fr;}.og2,.og3{grid-template-columns:1fr 1fr;}
  .peg{grid-template-columns:1fr 1fr;}.cg{grid-template-columns:1fr;}
  .exp-grid{grid-template-columns:1fr;}.tabs{gap:4px;}
  .tab-btn{padding:7px 11px;font-size:0.79rem;}
  .scan-btn{width:100%;justify-content:center;}
  .sr-name{max-width:90px;}
}
@media(max-width:420px){
  .sr-sc{display:none;}.og2{grid-template-columns:1fr;}
}

/* Glossary */
.gloss{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px;margin-bottom:13px;}
.gloss-title{font-size:0.82rem;font-weight:700;color:var(--teal);margin-bottom:9px;cursor:pointer;display:flex;justify-content:space-between;align-items:center;}
.gloss-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:6px;}
.gloss-item{background:rgba(0,0,0,0.2);border-radius:8px;padding:7px 10px;border-right:2px solid var(--blue);}
.gloss-en{font-size:0.75rem;font-weight:700;color:var(--blue);}
.gloss-ar{font-size:0.72rem;color:var(--txt);margin-top:2px;}
.gloss-desc{font-size:0.65rem;color:var(--mut);margin-top:2px;line-height:1.4;}
/* Sector Heat */
.heat-sec{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px;margin-bottom:13px;}
.heat-title{font-size:0.82rem;font-weight:700;color:var(--gold);margin-bottom:9px;}
.heat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:7px;}
.heat-card{border-radius:9px;padding:9px 11px;text-align:center;position:relative;overflow:hidden;}
.heat-name{font-size:0.78rem;font-weight:600;margin-bottom:4px;}
.heat-score{font-size:1.3rem;font-weight:700;}
.heat-bar{height:4px;border-radius:4px;margin-top:5px;}
.heat-info{font-size:0.62rem;color:rgba(255,255,255,0.6);margin-top:3px;}
</style>
"""

PAGE_JS = """
<script>
var aTab='tadawul',aFilt={tadawul:null,us:null,crypto:null};
function switchTab(t){
  aTab=t;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-c').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  document.getElementById('tc-'+t).classList.add('active');
}
function toggle(i){
  var det=document.getElementById('det'+i);
  var ch=document.getElementById('ch'+i);
  if(det.classList.contains('op')){det.classList.remove('op');ch.classList.remove('op');}
  else{det.classList.add('op');ch.classList.add('op');}
}
function filterChip(mkt,ft,el){
  var chips=document.querySelectorAll('#chips-'+mkt+' .chip');
  var rows=document.querySelectorAll('#list-'+mkt+' .srow');
  if(aFilt[mkt]===ft){
    aFilt[mkt]=null; chips.forEach(c=>c.classList.remove('af'));
    rows.forEach(r=>r.classList.remove('hidden'));
  } else {
    aFilt[mkt]=ft; chips.forEach(c=>c.classList.remove('af')); el.classList.add('af');
    rows.forEach(r=>{
      var show=false;
      if(ft==='buy'&&r.classList.contains('buy')) show=true;
      if(ft==='wait'&&r.classList.contains('wait')) show=true;
      if(ft==='avoid'&&r.classList.contains('avoid')) show=true;
      if(ft==='exp'&&r.querySelector('.exp-dot')) show=true;
      r.classList.toggle('hidden',!show);
    });
  }
}
function startScan(){
  document.getElementById('lo').classList.add('show');
  var lb={tadawul:'السوق السعودي',us:'السوق الأمريكي',crypto:'العملات'};
  document.getElementById('lm').textContent=lb[aTab]||aTab;
  fetch('/scan?market='+aTab).then(r=>r.json()).then(()=>chkStatus());
}
function chkStatus(){
  fetch('/status?market='+aTab).then(r=>r.json()).then(d=>{
    if(d.status==='done') window.location.reload();
    else setTimeout(chkStatus,2000);
  });
}
function copyVal(el){
  var t=el.textContent.trim();
  navigator.clipboard.writeText(t).then(()=>{
    el.classList.add('copied'); showToast('تم النسخ: '+t);
    setTimeout(()=>el.classList.remove('copied'),1300);
  });
}
function showToast(m){
  var t=document.getElementById('toast');
  t.textContent=m; t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),2000);
}
function addStock(){
  var code=document.getElementById('nCode').value.trim().toUpperCase();
  var name=document.getElementById('nName').value.trim();
  var mkt=document.getElementById('nMkt').value;
  if(!code||!name){showToast('أدخل الرمز والاسم');return;}
  fetch('/add_stock',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,name,market:mkt})})
  .then(r=>r.json()).then(d=>{if(d.ok){showToast('تمت الإضافة ✓');setTimeout(()=>location.reload(),800);}});
}
function uploadFile(){
  var file=document.getElementById('fileInput').files[0];
  var mkt=document.getElementById('nMkt').value;
  if(!file) return;
  var reader=new FileReader();
  reader.onload=function(e){
    var stocks=[];
    e.target.result.split('\\n').forEach(line=>{
      line=line.trim(); if(!line) return;
      var p=line.split(/\\s+/);
      if(p.length>=2) stocks.push({code:p[0].toUpperCase(),name:p.slice(1).join(' ')});
    });
    if(!stocks.length){showToast('ما في بيانات صحيحة');return;}
    fetch('/add_bulk',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({stocks,market:mkt})})
    .then(r=>r.json()).then(d=>{
      showToast('تمت إضافة '+d.count+' ✓');
      setTimeout(()=>location.reload(),1000);
    });
  };
  reader.readAsText(file);
}
function delStock(code,mkt){
  if(!confirm('حذف '+code+'؟')) return;
  fetch('/delete_stock',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,market:mkt})})
  .then(r=>r.json()).then(d=>{if(d.ok) location.reload();});
}
function exclStock(code){
  fetch('/exclude',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})})
  .then(r=>r.json()).then(d=>{if(d.ok) location.reload();});
}
function toggleGloss(){
  var b=document.getElementById('gloss-body');
  var c=document.getElementById('gloss-chev');
  if(b.style.display==='none'){b.style.display='block';c.textContent='▲';}
  else{b.style.display='none';c.textContent='▼';}
}
function inclStock(code){
  fetch('/include',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})})
  .then(r=>r.json()).then(d=>{if(d.ok) location.reload();});
}
</script>
"""

@app.route("/")
def index():
    custom=load_custom(); excl=custom.get("excluded",[])
    so,sl=is_us_session()
    sc="rgba(0,230,118,0.07)" if so else "rgba(255,61,87,0.05)"

    out=[PAGE_TOP]
    out.append('<body><div id="toast" class="toast"></div>')
    out.append('<div class="lo" id="lo"><div class="lr"></div>')
    out.append('<div class="lt">🔍 جاري مسح <span id="lm"></span>...</div>')
    out.append('<div class="ls2">أفضل 20 — 1-3 دقائق</div></div>')
    out.append('<div class="wrap">')
    out.append('<div class="hdr"><h1>⚡ جلال سكانر</h1>')
    out.append('<div class="sub">Jalal Scanner v3.1 Professional</div></div>')
    out.append('<div class="sess" style="background:'+sc+';border:1px solid var(--bdr);">🕐 '+sl+'</div>')
    out.append('<div class="lgd"><h4>📖 دليل سريع</h4><div class="lgd-g">')
    out.append('<div>⭐⭐⭐ استثمار — فوق الثلاثة — هدف أشهر</div>')
    out.append('<div>⭐⭐ سوينج — يومي+أسبوعي — هدف أسابيع</div>')
    out.append('<div>⭐ مضاربة — يومي فقط — هدف أيام</div>')
    out.append('<div>🟢 BUY = score 15+ وR:R 1.3+</div>')
    out.append('<div>🟡 BUY مشروط = score 13+ وR:R 1.0+</div>')
    out.append('<div>📦 Order Blocks — مناطق دعم مؤسسية</div>')
    out.append('<div>⚡ FVG — فجوات سعرية يرجع لها السعر</div>')
    out.append('<div>🚀 انفجار = حجم تداول شاذ × 3+</div>')
    out.append('<div>اضغط أي رقم في الأوامر لنسخه فوراً</div>')
    out.append('</div></div>')
    out.append('<div class="tabs">')
    out.append('<button class="tab-btn active" id="tab-tadawul" onclick="switchTab(\'tadawul\')">🇸🇦 تاسي</button>')
    out.append('<button class="tab-btn" id="tab-us" onclick="switchTab(\'us\')">🇺🇸 أمريكي</button>')
    out.append('<button class="tab-btn" id="tab-crypto" onclick="switchTab(\'crypto\')">💰 عملات</button>')
    out.append('</div>')

    # Custom panel
    out.append('<div class="cp"><h3>➕ إضافة أسهم</h3><div class="ci-row">')
    out.append('<input class="ci-inp" id="nCode" placeholder="الرمز" style="max-width:105px;">')
    out.append('<input class="ci-inp" id="nName" placeholder="الاسم">')
    out.append('<select class="ci-inp" id="nMkt" style="max-width:115px;">')
    out.append('<option value="tadawul">🇸🇦 تاسي</option>')
    out.append('<option value="us">🇺🇸 أمريكي</option>')
    out.append('<option value="crypto">💰 عملات</option>')
    out.append('</select><button class="add-btn" onclick="addStock()">➕</button></div>')
    out.append('<div class="file-row">')
    out.append('<label class="file-lbl" for="fileInput">📂 رفع ملف نصي</label>')
    out.append('<input type="file" id="fileInput" accept=".txt" style="display:none" onchange="uploadFile()">')
    out.append('<span class="file-hint">كل سطر: رمز اسم (مثال: 2222 أرامكو)</span>')
    out.append('</div>')
    if excl:
        out.append('<div style="margin-top:7px;font-size:0.66rem;color:var(--mut);">مستبعدون:</div><div>')
        for code in excl:
            out.append('<span class="ecl" onclick="inclStock(\''+code+'\')">'+code+' ↩</span>')
        out.append('</div>')
    out.append('</div>')


    # دليل المصطلحات
    out.append('<div class="gloss">')
    out.append('<div class="gloss-title" onclick="toggleGloss()">📖 دليل المصطلحات — اضغط للفتح <span id="gloss-chev">▼</span></div>')
    out.append('<div id="gloss-body" style="display:none"><div class="gloss-grid">')
    glossary = [
        ("RSI","مؤشر القوة النسبية","يقيس قوة حركة السعر. 40-70 = منطقة مثالية للدخول. أقل من 30 = ذروة بيع. أكثر من 70 = ذروة شراء"),
        ("MACD","تقاطع المتوسطات","يكشف تغيير الزخم. MACD > Signal = إشارة صاعدة ✅"),
        ("EMA","المتوسط المتحرك الأسي","يتبع السعر بسرعة أكبر من المتوسط العادي. فوقه = صاعد، تحته = هابط"),
        ("ADX","مؤشر قوة الاتجاه","يقيس مدى قوة الترند. أقل 20 = سوق عرضي (تجنب). 20-40 = ترند معقول ✅. 40+ = ترند قوي 🔥"),
        ("ATR","متوسط المدى الحقيقي","يقيس تقلب السعر اليومي. يُستخدم لحساب وقف الخسارة وجني الربح"),
        ("R:R","نسبة المخاطرة للعائد","1.3 يعني: لكل 1 ريال تخاطر به، تربح 1.3 ريال. كلما ارتفعت كلما كانت الصفقة أفضل"),
        ("Order Block","منطقة الدعم المؤسسي","المنطقة اللي دخل منها المال الكبير. لو رجع السعر لها وارتد = فرصة دخول قوية"),
        ("FVG","فجوة السعر — Fair Value Gap","فجوة تركها السعر بسرعة كبيرة. السعر غالباً يرجع يملأها قبل ما يكمل الاتجاه"),
        ("Trailing Stop","وقف خسارة متحرك","يتحرك مع السعر للأعلى تلقائياً. يحمي أرباحك ويترك الصفقة تكمل الصعود"),
        ("Break Even","نقطة التعادل","حرك وقف الخسارة لسعر الدخول بعد وصول TP1 — تأمن الصفقة بدون خسارة"),
        ("Partial Exit","خروج جزئي","بيع 50% عند TP1 و30% عند TP2، اترك 20% تجري مع الترند"),
        ("Relative Strength","القوة النسبية","مقارنة أداء السهم مع المؤشر. قوي↑ = السهم أفضل من السوق"),
        ("Liquidity Sweep","صيد السيولة","السعر يضرب قمة/قاع سابق ليصطاد أوامر الوقف قبل الحركة الحقيقية"),
        ("Sector Rotation","دوران القطاعات","انتقال الأموال من قطاع لآخر. القطاع اللي تدخله الأموال = الفرصة القادمة"),
        ("BUY مشروط","دخول مع تحفظ","الشروط الأساسية متحققة لكن فيه نقطة ضعف — راجع سبب التحفظ قبل الدخول"),
        ("Score 15+/20","إشارة قوية","15 نقطة فأكثر من 20 = معظم الشروط متحققة. أقل من 10 = ابتعد"),
    ]
    for en,ar,desc in glossary:
        out.append('<div class="gloss-item">')
        out.append('<div class="gloss-en">'+en+'</div>')
        out.append('<div class="gloss-ar">'+ar+'</div>')
        out.append('<div class="gloss-desc">'+desc+'</div>')
        out.append('</div>')
    out.append('</div></div></div>')


    # خريطة القطاعات (تاسي فقط)
    tadawul_data = scan_state["tadawul"]["data"] or []
    if tadawul_data:
        heat = get_sector_heat(tadawul_data)
        if heat:
            out.append('<div class="heat-sec">')
            out.append('<div class="heat-title">🌡️ خريطة حرارة القطاعات — أين تتجمع الفلوس؟</div>')
            out.append('<div class="heat-grid">')
            for sector, info in heat.items():
                avg = info["avg"]
                buy = info["buy"]
                cnt = info["count"]
                # لون بناءً على المتوسط
                if avg >= 14:
                    bg = "rgba(0,230,118,0.12)"; bc = "rgba(0,230,118,0.4)"; sc = "var(--green)"
                    label = "🔥 ساخن"
                elif avg >= 10:
                    bg = "rgba(255,179,0,0.1)";  bc = "rgba(255,179,0,0.35)"; sc = "var(--yel)"
                    label = "👀 محايد"
                else:
                    bg = "rgba(255,61,87,0.08)"; bc = "rgba(255,61,87,0.3)"; sc = "var(--red)"
                    label = "❄️ بارد"
                pct = min(int(avg/20*100), 100)
                out.append('<div class="heat-card" style="background:'+bg+';border:1px solid '+bc+';">')
                out.append('<div class="heat-name">'+sector+'</div>')
                out.append('<div class="heat-score" style="color:'+sc+'">'+str(avg)+'</div>')
                out.append('<div class="heat-bar" style="background:'+sc+';width:'+str(pct)+'%;margin:0 auto;"></div>')
                out.append('<div class="heat-info">'+label+' | '+str(buy)+' BUY من '+str(cnt)+'</div>')
                out.append('</div>')
            out.append('</div></div>')

    for mkt,label,flag in [("tadawul","السوق السعودي","🇸🇦"),("us","السوق الأمريكي","🇺🇸"),("crypto","العملات","💰")]:
        data=scan_state[mkt]["data"] or []
        last=scan_state[mkt]["last_scan"] or ""
        acls=" active" if mkt=="tadawul" else ""
        out.append('<div class="tab-c'+acls+'" id="tc-'+mkt+'">')
        out.append('<div style="text-align:center;padding:5px 0 11px;">')
        out.append('<div class="ls">'+("آخر مسح: "+last+" (أفضل 20)" if last else "لم يتم المسح بعد")+'</div>')
        out.append('<button class="scan-btn" onclick="startScan()">🔍 مسح '+flag+' '+label+'</button>')
        out.append('</div>')

        if data:
            buy_l=[s for s in data if s["verdict"]=="BUY"]
            wait_l=[s for s in data if s["verdict"]=="WAIT"]
            avoid_l=[s for s in data if s["verdict"]=="AVOID"]
            exp_l=[s for s in data if s.get("exp")]
            cond_l=[s for s in buy_l if "مشروط" in s["bt"]]
            full_l=[s for s in buy_l if "مشروط" not in s["bt"]]

            out.append('<div class="chips" id="chips-'+mkt+'">')
            out.append('<div class="chip co" onclick="filterChip(\''+mkt+'\',\'all\',this)"><div class="num">'+str(len(data))+'</div><div class="lbl">الكل</div></div>')
            out.append('<div class="chip cg" onclick="filterChip(\''+mkt+'\',\'buy\',this)"><div class="num">'+str(len(buy_l))+'</div><div class="lbl">🟢 BUY</div></div>')
            out.append('<div class="chip cy" onclick="filterChip(\''+mkt+'\',\'wait\',this)"><div class="num">'+str(len(wait_l))+'</div><div class="lbl">🟡 WAIT</div></div>')
            out.append('<div class="chip cr" onclick="filterChip(\''+mkt+'\',\'avoid\',this)"><div class="num">'+str(len(avoid_l))+'</div><div class="lbl">🔴 AVOID</div></div>')
            out.append('<div class="chip co" onclick="filterChip(\''+mkt+'\',\'exp\',this)"><div class="num">'+str(len(exp_l))+'</div><div class="lbl">🚀 انفجار</div></div>')
            out.append('</div>')

            if exp_l:
                out.append('<div class="exp-sec"><div class="exp-ttl">🚀 تنبيهات الانفجار</div><div class="exp-grid">')
                for s in exp_l:
                    sfx=".SR" if s["market"]=="tadawul" else ""
                    out.append('<div class="exp-card"><div><div class="exp-nm">'+s["name"]+'</div><div class="exp-cd">'+s["code"]+sfx+'</div></div>')
                    out.append('<div class="exp-px">'+str(s["price"])+'</div>')
                    out.append('<div class="exp-rt">🚀 ×'+str(s["vr"])+'</div>')
                    out.append('<span style="font-size:0.68rem">'+s["bt"]+'</span></div>')
                out.append('</div></div>')

            out.append('<div class="slist" id="list-'+mkt+'">')
            for i,s in enumerate(data):
                out.append(row_html(s,i))
            out.append('</div>')
        else:
            out.append('<div class="nr"><div style="font-size:2rem">📡</div><div style="margin-top:6px">اضغط "مسح" لتحليل '+label+'</div></div>')
        out.append('</div>')

    out.append('</div>'+PAGE_JS+'</body></html>')
    return "".join(out)

@app.route("/scan")
def scan():
    m=request.args.get("market","tadawul")
    if scan_state[m]["status"]=="scanning": return jsonify({"status":"already_running"})
    t=threading.Thread(target=run_scan,args=(m,)); t.daemon=True; t.start()
    return jsonify({"status":"started"})

@app.route("/status")
def status():
    m=request.args.get("market","tadawul")
    return jsonify({"status":scan_state[m]["status"]})

@app.route("/add_stock",methods=["POST"])
def add_stock():
    d=request.get_json(); code=d.get("code","").strip(); name=d.get("name","").strip(); mkt=d.get("market","tadawul")
    if not code or not name: return jsonify({"ok":False})
    c=load_custom(); c.setdefault(mkt,{})[code]=name
    if code in c.get("excluded",[]): c["excluded"].remove(code)
    save_custom(c); return jsonify({"ok":True})

@app.route("/add_bulk",methods=["POST"])
def add_bulk():
    d=request.get_json(); stocks=d.get("stocks",[]); mkt=d.get("market","tadawul")
    c=load_custom(); count=0
    for item in stocks:
        code=item.get("code","").strip(); name=item.get("name","").strip()
        if code and name:
            c.setdefault(mkt,{})[code]=name
            if code in c.get("excluded",[]): c["excluded"].remove(code)
            count+=1
    save_custom(c); return jsonify({"ok":True,"count":count})

@app.route("/delete_stock",methods=["POST"])
def delete_stock():
    d=request.get_json(); code=d.get("code","").strip(); mkt=d.get("market","tadawul")
    c=load_custom()
    if code in c.get(mkt,{}): del c[mkt][code]
    save_custom(c); return jsonify({"ok":True})

@app.route("/exclude",methods=["POST"])
def exclude():
    code=request.get_json().get("code",""); c=load_custom()
    if code not in c.get("excluded",[]): c.setdefault("excluded",[]).append(code)
    save_custom(c); return jsonify({"ok":True})

@app.route("/include",methods=["POST"])
def include():
    code=request.get_json().get("code",""); c=load_custom()
    if code in c.get("excluded",[]): c["excluded"].remove(code)
    save_custom(c); return jsonify({"ok":True})

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    print("="*50); print("  ⚡ جلال سكانر v3.1"); print("  http://localhost:5000"); print("="*50)
    if port==5000: threading.Timer(1.5,lambda:webbrowser.open("http://localhost:5000")).start()
    app.run(host="0.0.0.0",port=port,debug=False)
