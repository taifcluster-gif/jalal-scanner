#!/usr/bin/env python3
# جلال سكانر v2.2
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings, threading, webbrowser, os, json
warnings.filterwarnings("ignore")

app = Flask(__name__)

def load_binance_keys():
    keys = {"API_KEY":"","SECRET_KEY":""}
    kf = os.path.join(os.path.dirname(os.path.abspath(__file__)),"binance_keys")
    if os.path.exists(kf):
        with open(kf,"r") as f:
            for line in f:
                if "=" in line:
                    k,v = line.strip().split("=",1)
                    keys[k.strip()] = v.strip()
    return keys

BINANCE_KEYS = load_binance_keys()
CUSTOM_FILE  = os.path.join(os.path.dirname(os.path.abspath(__file__)),"custom_stocks.json")

def load_custom():
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return {"tadawul":{},"us":{},"crypto":{},"excluded":[]}

def save_custom(data):
    with open(CUSTOM_FILE,"w",encoding="utf-8") as f:
        json.dump(data,f,ensure_ascii=False,indent=2)

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
    "COST":"Costco","HD":"Home Depot",
}
DEFAULT_CRYPTO = {
    "BTCUSDT":"Bitcoin","ETHUSDT":"Ethereum","BNBUSDT":"BNB",
    "SOLUSDT":"Solana","XRPUSDT":"Ripple","ADAUSDT":"Cardano",
}

scan_state = {
    "tadawul":{"data":None,"last_scan":None,"status":"idle"},
    "us":     {"data":None,"last_scan":None,"status":"idle"},
    "crypto": {"data":None,"last_scan":None,"status":"idle"},
}

def get_df(ticker,period,interval):
    try:
        df=yf.download(ticker,period=period,interval=interval,progress=False,auto_adjust=True)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns,pd.MultiIndex): df.columns=df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

def ema(s,p): return s.ewm(span=p,adjust=False).mean()
def sma(s,p): return s.rolling(p).mean()

def rsi_calc(s,p=14):
    d=s.diff(); g=d.clip(lower=0).rolling(p).mean(); l=(-d.clip(upper=0)).rolling(p).mean()
    return 100-(100/(1+g/l.replace(0,np.nan)))

def macd_calc(s):
    ml=ema(s,12)-ema(s,26); sg=ema(ml,9); return ml,sg,ml-sg

def adx_calc(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    at=tr.rolling(p).mean()
    up,dn=h.diff(),-l.diff()
    dp=pd.Series(np.where((up>dn)&(up>0),up,0.),index=h.index).rolling(p).mean()
    dm=pd.Series(np.where((dn>up)&(dn>0),dn,0.),index=h.index).rolling(p).mean()
    dip,dim=100*dp/at,100*dm/at
    dx=100*(dip-dim).abs()/(dip+dim).replace(0,np.nan)
    return dx.rolling(p).mean()

def stoch_calc(h,l,c,k=14,d=3):
    kv=100*(c-l.rolling(k).min())/(h.rolling(k).max()-l.rolling(k).min()).replace(0,np.nan)
    return kv,kv.rolling(d).mean()

def atr_calc(h,l,c,p=14):
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    return tr.rolling(p).mean()

def estimate_duration(trend,atr_v,price,tp1):
    if atr_v<=0: return "غير محدد"
    days=max(1,round((tp1-price)/atr_v))
    if trend=="استثمار": return f"{days}-{days*2} يوم"
    elif trend=="سوينج": return f"{days}-{days+3} أيام"
    elif trend=="مضاربة": return f"{max(1,days-1)}-{days+2} أيام"
    return "غير محدد"

def analyze_stock(code,name,market="tadawul"):
    try:
        ticker=code+".SR" if market=="tadawul" else code
        df_d=get_df(ticker,"2y","1d"); df_w=get_df(ticker,"5y","1wk"); df_m=get_df(ticker,"10y","1mo")
        if df_d.empty or len(df_d)<50: return None
        price=float(df_d["Close"].iloc[-1])
        ema20_d=float(ema(df_d["Close"],20).iloc[-1])
        ema20_w=float(ema(df_w["Close"],20).iloc[-1]) if len(df_w)>=20 else None
        ema20_m=float(ema(df_m["Close"],20).iloc[-1]) if len(df_m)>=20 else None
        ad=price>ema20_d; aw=price>ema20_w if ema20_w else False; am=price>ema20_m if ema20_m else False
        if ad and aw and am:   trend,stars="استثمار",3
        elif ad and aw:        trend,stars="سوينج",2
        elif ad:               trend,stars="مضاربة",1
        elif not ad and not aw and not am: trend,stars="تجنب",0
        else:                  trend,stars="انتظر",0
        d=df_d.copy()
        e20=ema(d["Close"],20);e50=ema(d["Close"],50);e200=ema(d["Close"],200)
        rv=rsi_calc(d["Close"]);ml,sg,mh=macd_calc(d["Close"])
        adv=adx_calc(d["High"],d["Low"],d["Close"])
        sk,sdv=stoch_calc(d["High"],d["Low"],d["Close"])
        va=sma(d["Volume"],20);bm=sma(d["Close"],20)
        bw=(4*d["Close"].rolling(20).std())/bm
        av=atr_calc(d["High"],d["Low"],d["Close"])
        c1=price>float(e20.iloc[-1]);c2=float(e20.iloc[-1])>float(e50.iloc[-1])
        c3=price>float(e200.iloc[-1]);c4=float(ml.iloc[-1])>float(sg.iloc[-1])
        c5=float(mh.iloc[-1])>float(mh.iloc[-2]);c6=40<=float(rv.iloc[-1])<=70
        c7=float(rv.iloc[-1])>float(rv.iloc[-2]);c8=float(adv.iloc[-1])>20
        c9=float(sk.iloc[-1])>20 and float(sk.iloc[-1])>float(sdv.iloc[-1])
        c10=float(d["Volume"].iloc[-1])>float(va.iloc[-1])*1.2
        c11=price>float(bm.iloc[-1]);c12=float(bw.iloc[-1])>float(bw.iloc[-2])
        vol_ratio=float(d["Volume"].iloc[-1])/float(va.iloc[-1]) if float(va.iloc[-1])>0 else 1
        explosion=vol_ratio>=3.0
        score=(2 if c1 else 0)+(2 if c2 else 0)+(2 if c3 else 0)+(2 if c4 else 0)+\
              (1 if c5 else 0)+(2 if c6 else 0)+(1 if c7 else 0)+(2 if c8 else 0)+\
              (2 if c9 else 0)+(2 if c10 else 0)+(1 if c11 else 0)+(1 if c12 else 0)
        atr_v=float(av.iloc[-1])
        entry       = round(price,3)
        limit_buy   = round(price*0.995,3)
        # OCO — جني ربح
        tp1_stop    = round(price+atr_v*1.8,3)   # سعر الإيقاف TP
        tp1_limit   = round(price+atr_v*2.0,3)   # سعر الأمر TP
        tp2_stop    = round(price+atr_v*3.8,3)
        tp2_limit   = round(price+atr_v*4.0,3)
        # OCO — وقف الخسارة
        sl_stop     = round(price-atr_v*1.3,3)   # سعر الإيقاف SL
        sl_limit    = round(price-atr_v*1.5,3)   # سعر الأمر SL
        rr          = round((tp1_limit-entry)/max(entry-sl_limit,0.001),2)
        pct_sl      = round((entry-sl_limit)/entry*100,2)
        pct_tp      = round((tp1_limit-entry)/entry*100,2)
        # Trailing
        trail_pct   = round((atr_v*1.5/price)*100,2)
        trail_gap   = round(atr_v*0.3,3)          # الفارق السعري
        duration    = estimate_duration(trend,atr_v,price,tp1_limit)
        if score>=15 and rr>=1.5: verdict,priority="BUY",1
        elif score>=10:            verdict,priority="WAIT",2
        else:                      verdict,priority="AVOID",3
        conds=[
            {"name":"Close > EMA20","ok":c1,"w":2},{"name":"EMA20 > EMA50","ok":c2,"w":2},
            {"name":"Close > EMA200","ok":c3,"w":2},{"name":"MACD > Signal","ok":c4,"w":2},
            {"name":"MACD Rising","ok":c5,"w":1},{"name":"RSI (40-70)","ok":c6,"w":2},
            {"name":"RSI Rising","ok":c7,"w":1},{"name":"ADX > 20","ok":c8,"w":2},
            {"name":"Stoch > 20","ok":c9,"w":2},{"name":"Vol > Avg×1.2","ok":c10,"w":2},
            {"name":"Above BB Mid","ok":c11,"w":1},{"name":"BB Expanding","ok":c12,"w":1},
        ]
        return {
            "code":code,"name":name,"price":entry,"market":market,
            "above_daily":ad,"above_weekly":aw,"above_monthly":am,
            "trend":trend,"stars":stars,"score":score,"score_pct":round(score/20*100),
            "verdict":verdict,"priority":priority,
            "entry":entry,"limit_buy":limit_buy,
            "tp1_stop":tp1_stop,"tp1_limit":tp1_limit,
            "tp2_stop":tp2_stop,"tp2_limit":tp2_limit,
            "sl_stop":sl_stop,"sl_limit":sl_limit,
            "rr":rr,"pct_sl":pct_sl,"pct_tp":pct_tp,
            "trail_pct":trail_pct,"trail_gap":trail_gap,
            "duration":duration,
            "rsi":round(float(rv.iloc[-1]),1),"adx":round(float(adv.iloc[-1]),1),
            "vol_ratio":round(vol_ratio,1),"explosion":explosion,
            "conditions":conds,"conds_ok":sum(1 for c in conds if c["ok"]),
            "is_custom":False,
        }
    except: return None

def get_crypto_price(symbol):
    try:
        api_key=BINANCE_KEYS.get("API_KEY","")
        if api_key:
            import urllib.request,json as j
            url=f"https://api.binance.com/api/v3/ticker/24hr?symbol={symbol}"
            req=urllib.request.Request(url,headers={"X-MBX-APIKEY":api_key})
            with urllib.request.urlopen(req,timeout=5) as r:
                data=j.loads(r.read())
                return {"price":float(data["lastPrice"]),"change_pct":float(data["priceChangePercent"]),
                        "high":float(data["highPrice"]),"low":float(data["lowPrice"])}
    except: pass
    try:
        yf_sym=symbol.replace("USDT","-USD")
        df=get_df(yf_sym,"5d","1h")
        if not df.empty:
            p=float(df["Close"].iloc[-1]); p2=float(df["Close"].iloc[-24]) if len(df)>=24 else p
            return {"price":p,"change_pct":round((p-p2)/p2*100,2),"high":float(df["High"].max()),"low":float(df["Low"].min())}
    except: pass
    return None

def analyze_crypto(symbol,name):
    try:
        info=get_crypto_price(symbol)
        if not info: return None
        price=info["price"]; change=info["change_pct"]
        yf_sym=symbol.replace("USDT","-USD")
        df_d=get_df(yf_sym,"2y","1d")
        if df_d.empty or len(df_d)<50: return None
        ema20_d=float(ema(df_d["Close"],20).iloc[-1])
        df_w=get_df(yf_sym,"5y","1wk")
        ema20_w=float(ema(df_w["Close"],20).iloc[-1]) if len(df_w)>=20 else None
        ad=price>ema20_d; aw=price>ema20_w if ema20_w else False
        if ad and aw:   trend,stars="سوينج",2
        elif ad:        trend,stars="مضاربة",1
        else:           trend,stars="تجنب",0
        d=df_d.copy()
        rv=rsi_calc(d["Close"]); ml,sg,mh=macd_calc(d["Close"])
        av=atr_calc(d["High"],d["Low"],d["Close"])
        atr_v=float(av.iloc[-1])
        entry=round(price,4); limit_buy=round(price*0.995,4)
        tp1_stop=round(price+atr_v*1.8,4); tp1_limit=round(price+atr_v*2.0,4)
        tp2_stop=round(price+atr_v*3.8,4); tp2_limit=round(price+atr_v*4.0,4)
        sl_stop=round(price-atr_v*1.3,4); sl_limit=round(price-atr_v*1.5,4)
        rr=round((tp1_limit-entry)/max(entry-sl_limit,0.001),2)
        pct_sl=round((entry-sl_limit)/entry*100,2); pct_tp=round((tp1_limit-entry)/entry*100,2)
        trail_pct=round((atr_v*1.5/price)*100,2); trail_gap=round(atr_v*0.3,4)
        score=10 if (ad and aw) else (6 if ad else 3)
        if score>=8 and rr>=1.5: verdict,priority="BUY",1
        elif score>=5:            verdict,priority="WAIT",2
        else:                     verdict,priority="AVOID",3
        return {
            "code":symbol,"name":name,"price":entry,"market":"crypto",
            "change_pct":change,"above_daily":ad,"above_weekly":aw,"above_monthly":False,
            "trend":trend,"stars":stars,"score":score,"score_pct":round(score/20*100),
            "verdict":verdict,"priority":priority,
            "entry":entry,"limit_buy":limit_buy,
            "tp1_stop":tp1_stop,"tp1_limit":tp1_limit,
            "tp2_stop":tp2_stop,"tp2_limit":tp2_limit,
            "sl_stop":sl_stop,"sl_limit":sl_limit,
            "rr":rr,"pct_sl":pct_sl,"pct_tp":pct_tp,
            "trail_pct":trail_pct,"trail_gap":trail_gap,
            "duration":estimate_duration(trend,atr_v,price,tp1_limit),
            "rsi":round(float(rv.iloc[-1]),1),"adx":0,
            "vol_ratio":1,"explosion":change>10,
            "conditions":[],"conds_ok":0,"is_custom":False,
        }
    except: return None

def run_scan(market):
    scan_state[market]["status"]="scanning"
    custom=load_custom(); excluded=custom.get("excluded",[])
    results=[]
    if market=="tadawul":
        stocks={**DEFAULT_TADAWUL,**custom.get("tadawul",{})}
        for code,name in stocks.items():
            if code in excluded: continue
            r=analyze_stock(code,name,"tadawul")
            if r:
                r["is_custom"]=code in custom.get("tadawul",{})
                results.append(r)
    elif market=="us":
        stocks={**DEFAULT_US,**custom.get("us",{})}
        for code,name in stocks.items():
            if code in excluded: continue
            r=analyze_stock(code,name,"us")
            if r:
                r["is_custom"]=code in custom.get("us",{})
                results.append(r)
    elif market=="crypto":
        stocks={**DEFAULT_CRYPTO,**custom.get("crypto",{})}
        for code,name in stocks.items():
            if code in excluded: continue
            r=analyze_crypto(code,name)
            if r:
                r["is_custom"]=code in custom.get("crypto",{})
                results.append(r)
    results.sort(key=lambda x:(x["priority"],-x["score"],-x["stars"]))
    scan_state[market]["data"]=results
    scan_state[market]["last_scan"]=datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state[market]["status"]="done"

def cv(val):
    """Copy value button"""
    return '<span class="cv" onclick="copyVal(this)">'+str(val)+'</span>'

def render_card(s,idx):
    vl=s['verdict'].lower()
    var="🟢 اشتري" if s['verdict']=='BUY' else ("🟡 انتظر" if s['verdict']=='WAIT' else "🔴 تجنب")
    stars_html="⭐"*s['stars']
    trend_tips={"استثمار":"فوق الثلاثة — هدف أشهر","سوينج":"يومي+أسبوعي — هدف أسابيع","مضاربة":"يومي فقط — هدف أيام","انتظر":"غير مكتملة — لا تدخل","تجنب":"تحت المتوسطات — ابتعد"}
    tip=trend_tips.get(s['trend'],"")
    dc="chip-yes" if s['above_daily'] else "chip-no"
    wc="chip-yes" if s['above_weekly'] else "chip-no"
    mc="chip-yes" if s['above_monthly'] else "chip-no"
    sp=str(s['score_pct'])
    exp_badge='<span class="exp-badge">🚀 ×'+str(s['vol_ratio'])+'</span>' if s.get('explosion') else ''
    cust_badge='<span class="cust-badge">✏️</span>' if s.get('is_custom') else ''
    chg_html=""
    if "change_pct" in s:
        chg=s['change_pct']; cls="pos" if chg>=0 else "neg"
        chg_html=f'<span class="{cls}">{"+" if chg>=0 else ""}{chg}%</span>'
    sfx=".SR" if s['market']=="tadawul" else ""
    conds_html="".join(
        '<div class="ci '+ ("co" if c['ok'] else "cf") +'"><div class="cd '+ ("dok" if c['ok'] else "dfail") +'"></div><span>'+c["name"]+'</span><span class="cw">★'+str(c["w"])+'</span></div>'
        for c in s['conditions']
    )
    del_btn='<button class="del-btn" onclick="deleteStock(\''+s['code']+'\',\''+s['market']+'\')">🗑 حذف من القائمة</button>'
    excl_btn='<button class="excl-btn" onclick="excludeStock(\''+s['code']+'\')">⊘ استبعاد مؤقت</button>'

    return (
        '<div class="sc '+vl+'" style="animation-delay:'+str(idx*0.04)+'s">'
        # Header
        '<div class="ch">'
        '<div><div class="sn">'+s['name']+' '+exp_badge+' '+cust_badge+'</div>'
        '<div class="scode">'+s['code']+sfx+'</div></div>'
        '<div style="text-align:left"><div class="sp">'+str(s['price'])+'</div>'+chg_html+'</div>'
        '</div>'
        # Verdict + stars
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;">'
        '<span class="vb vb-'+vl+'">'+var+'</span>'
        '<span class="stars" title="'+tip+'">'+stars_html+' <span class="tn">'+s['trend']+'</span></span>'
        '</div>'
        # Trend chips
        '<div class="tr">'
        '<span class="tl">EMA20:</span>'
        '<span class="tc '+dc+'" title="السعر فوق المتوسط اليومي">يومي '+("✓" if s['above_daily'] else "✗")+'</span>'
        '<span class="tc '+wc+'" title="السعر فوق المتوسط الأسبوعي">أسبوعي '+("✓" if s['above_weekly'] else "✗")+'</span>'
        '<span class="tc '+mc+'" title="السعر فوق المتوسط الشهري">شهري '+("✓" if s['above_monthly'] else "✗")+'</span>'
        '<span class="rsi-b" title="RSI: 40-70 مثالي">RSI:'+str(s['rsi'])+'</span>'
        '</div>'
        # Score
        '<div class="ss">'
        '<div class="sh"><span title="النقاط من 20 — 15+ قوي | 10-14 متوسط | أقل ضعيف">JRF Score</span>'
        '<span class="sn2">'+str(s['score'])+'/20 <span class="sp2">'+sp+'%</span></span></div>'
        '<div class="st"><div class="sf fill-'+vl+'" style="width:'+sp+'px;max-width:100%;"></div></div>'
        '<div class="sl2"><span style="color:var(--red)">0-9 ضعيف</span><span style="color:var(--yellow)">10-14 متوسط</span><span style="color:var(--green)">15+ قوي</span></div>'
        '</div>'

        # ── أمر الشراء ──
        '<div class="os">'
        '<div class="ot">🛒 أمر الشراء</div>'
        '<div class="og1">'
        '<div class="oi"><div class="ol">سعر الأمر (Limit)</div><div class="ov entry">'+cv(s['limit_buy'])+'</div><div class="oh">اضغط للنسخ</div></div>'
        '</div>'
        '</div>'

        # ── جني الربح ──
        '<div class="os">'
        '<div class="ot">💰 جني الربح (Take Profit)</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov tp">'+cv(s['tp1_stop'])+'</div><div class="oh">Trigger Price</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov tp">'+cv(s['tp1_limit'])+'</div><div class="oh pos">+'+str(s['pct_tp'])+'%</div></div>'
        '</div>'
        '<div class="og2" style="margin-top:6px;">'
        '<div class="oi"><div class="ol">إيقاف 2</div><div class="ov tp">'+cv(s['tp2_stop'])+'</div><div class="oh">TP2 Trigger</div></div>'
        '<div class="oi"><div class="ol">أمر 2</div><div class="ov tp">'+cv(s['tp2_limit'])+'</div><div class="oh pos">+'+str(round(s['pct_tp']*2,2))+'%</div></div>'
        '</div>'
        '</div>'

        # ── وقف الخسارة ──
        '<div class="os os-sl">'
        '<div class="ot">🛡 وقف الخسارة (Stop Loss)</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov sl">'+cv(s['sl_stop'])+'</div><div class="oh">Trigger Price</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov sl">'+cv(s['sl_limit'])+'</div><div class="oh neg">-'+str(s['pct_sl'])+'%</div></div>'
        '</div>'
        '<div class="rr-row">'
        '<span>نسبة R:R: <strong style="color:var(--gold)">'+str(s['rr'])+'</strong></span>'
        '<span>⏱ '+s['duration']+'</span>'
        '</div>'
        '</div>'

        # ── أمر التتبع ──
        '<div class="os os-trail">'
        '<div class="ot">🚀 أمر التتبع (Trailing Stop)</div>'
        '<div class="og3">'
        '<div class="oi"><div class="ol">نوع التتبع</div><div class="ov" style="color:var(--teal)">نسبة مئوية</div><div class="oh">Percentage</div></div>'
        '<div class="oi"><div class="ol">المبلغ / النسبة</div><div class="ov sl">'+cv(str(s['trail_pct'])+'%')+'</div><div class="oh">أدخل هذه النسبة</div></div>'
        '<div class="oi"><div class="ol">الفارق السعري</div><div class="ov" style="color:var(--muted)">'+cv(s['trail_gap'])+'</div><div class="oh">Price Gap</div></div>'
        '</div>'
        '<div class="trail-note">💡 سعر الدخول: '+str(s['entry'])+'  |  الهدف الأقصى: '+str(s['tp2_limit'])+'</div>'
        '</div>'

        # Conditions
        +(
            '<details><summary class="cb">📊 الشروط التفصيلية ('+str(s['conds_ok'])+'/12)</summary>'
            '<div class="cleg">★★ = 2 نقطة  |  ★ = 1 نقطة</div>'
            '<div class="cg">'+conds_html+'</div>'
            '</details>'
            if conds_html else ''
        )

        # أزرار الإدارة
        +'<div class="btn-row">'
        +(del_btn if s.get('is_custom') else '')
        +excl_btn
        +'</div>'
        +'</div>'
    )

CSS = """<style>
:root{--bg:#0a0e1a;--card:#131d35;--border:#1e2d4a;--gold:#f5c518;--green:#00e676;--red:#ff3d57;--yellow:#ffb300;--blue:#2979ff;--teal:#00bcd4;--text:#e8eaf6;--muted:#7986cb;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'IBM Plex Sans Arabic',sans-serif;min-height:100vh;}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse at 20% 50%,rgba(41,121,255,0.05),transparent 60%),radial-gradient(ellipse at 80% 20%,rgba(0,188,212,0.05),transparent 60%);pointer-events:none;z-index:0;}
.wrapper{position:relative;z-index:1;max-width:1400px;margin:0 auto;padding:20px;}
.header{text-align:center;padding:26px 20px 16px;border-bottom:1px solid var(--border);margin-bottom:18px;}
.header h1{font-size:clamp(1.5rem,4vw,2.5rem);font-weight:700;background:linear-gradient(135deg,var(--gold),var(--teal));-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:2px;margin-bottom:4px;}
.sub{color:var(--muted);font-size:0.8rem;}.ls{color:var(--muted);font-size:0.74rem;margin-top:3px;}
.tabs{display:flex;gap:7px;justify-content:center;margin-bottom:18px;flex-wrap:wrap;}
.tab-btn{padding:9px 24px;border-radius:50px;border:1px solid var(--border);background:transparent;color:var(--muted);font-family:inherit;font-size:0.9rem;cursor:pointer;transition:all 0.3s;}
.tab-btn:hover{border-color:var(--blue);color:var(--text);}
.tab-btn.active{background:linear-gradient(135deg,#1a3a6e,#0d2044);border-color:var(--blue);color:var(--text);}
.tab-content{display:none;}.tab-content.active{display:block;}
.scan-btn{display:inline-flex;align-items:center;gap:7px;background:linear-gradient(135deg,#1a3a6e,#0d2044);border:1px solid var(--blue);color:var(--text);padding:10px 26px;border-radius:50px;font-size:0.92rem;font-family:inherit;cursor:pointer;margin-top:12px;transition:all 0.3s;}
.scan-btn:hover{transform:translateY(-2px);box-shadow:0 8px 28px rgba(41,121,255,0.3);}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:0.4}}
.pulse{animation:pulse 1.5s infinite;}
.stats-bar{display:flex;gap:9px;flex-wrap:wrap;margin-bottom:18px;}
.stat-card{flex:1;min-width:85px;background:var(--card);border:1px solid var(--border);border-radius:12px;padding:11px 13px;text-align:center;}
.stat-card .num{font-size:1.65rem;font-weight:700;}.stat-card .lbl{font-size:0.68rem;color:var(--muted);margin-top:2px;}
.stat-card.green{border-color:var(--green)}.stat-card.green .num{color:var(--green)}
.stat-card.yellow{border-color:var(--yellow)}.stat-card.yellow .num{color:var(--yellow)}
.stat-card.red{border-color:var(--red)}.stat-card.red .num{color:var(--red)}
.stat-card.gold{border-color:var(--gold)}.stat-card.gold .num{color:var(--gold)}
/* Explosion Section */
.exp-section{background:rgba(245,197,24,0.06);border:1px solid rgba(245,197,24,0.3);border-radius:14px;padding:14px;margin-bottom:20px;}
.exp-section-title{color:var(--gold);font-size:0.95rem;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:8px;}
.exp-mini-card{background:rgba(0,0,0,0.3);border:1px solid rgba(245,197,24,0.25);border-radius:10px;padding:10px 14px;display:flex;align-items:center;justify-content:space-between;gap:10px;}
.exp-mini-name{font-weight:600;font-size:0.9rem;}
.exp-mini-code{color:var(--muted);font-size:0.7rem;}
.exp-mini-price{color:var(--gold);font-weight:700;}
.exp-mini-ratio{background:rgba(245,197,24,0.15);color:var(--yellow);border:1px solid rgba(245,197,24,0.3);padding:3px 10px;border-radius:10px;font-size:0.75rem;font-weight:600;}
.exp-mini-verdict{font-size:0.75rem;}
.exp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;}
/* Section titles */
.sec-t{font-size:0.93rem;font-weight:600;padding:7px 13px;border-radius:8px;margin:20px 0 11px;display:flex;align-items:center;gap:7px;}
.sec-buy{background:rgba(0,230,118,0.1);border-right:4px solid var(--green);color:var(--green);}
.sec-wait{background:rgba(255,179,0,0.1);border-right:4px solid var(--yellow);color:var(--yellow);}
.sec-avoid{background:rgba(255,61,87,0.1);border-right:4px solid var(--red);color:var(--red);}
/* Cards */
.cards-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:15px;}
@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
.sc{background:var(--card);border:1px solid var(--border);border-radius:18px;padding:16px;position:relative;overflow:hidden;transition:transform 0.3s,box-shadow 0.3s;animation:fadeUp 0.5s ease both;}
.sc:hover{transform:translateY(-3px);box-shadow:0 12px 32px rgba(0,0,0,0.4);}
.sc.buy{border-color:rgba(0,230,118,0.4)}.sc.wait{border-color:rgba(255,179,0,0.3)}.sc.avoid{border-color:rgba(255,61,87,0.3)}
.sc::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:18px 18px 0 0;}
.sc.buy::before{background:linear-gradient(90deg,var(--green),var(--teal))}
.sc.wait::before{background:linear-gradient(90deg,var(--yellow),var(--gold))}
.sc.avoid::before{background:linear-gradient(90deg,var(--red),#ff6b35)}
.ch{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:11px;}
.sn{font-size:1.05rem;font-weight:700;}.scode{font-size:0.68rem;color:var(--muted);margin-top:2px;}
.sp{font-size:1.3rem;font-weight:700;color:var(--gold);}
.vb{display:inline-block;padding:3px 11px;border-radius:14px;font-size:0.74rem;font-weight:600;margin-bottom:9px;}
.vb-buy{background:rgba(0,230,118,0.15);color:var(--green);border:1px solid var(--green);}
.vb-wait{background:rgba(255,179,0,0.15);color:var(--yellow);border:1px solid var(--yellow);}
.vb-avoid{background:rgba(255,61,87,0.15);color:var(--red);border:1px solid var(--red);}
.tr{display:flex;gap:5px;align-items:center;margin-bottom:10px;flex-wrap:wrap;}
.tc{padding:2px 7px;border-radius:7px;font-size:0.67rem;font-weight:600;cursor:help;}
.chip-yes{background:rgba(0,230,118,0.15);color:var(--green);border:1px solid rgba(0,230,118,0.3);}
.chip-no{background:rgba(255,61,87,0.1);color:var(--red);border:1px solid rgba(255,61,87,0.2);}
.tl{font-size:0.74rem;color:var(--muted);}.tn{color:var(--muted);font-size:0.7rem;}
.rsi-b{background:rgba(121,134,203,0.12);color:var(--muted);border:1px solid rgba(121,134,203,0.25);padding:2px 7px;border-radius:7px;font-size:0.66rem;margin-right:auto;}
.ss{margin-bottom:11px;}
.sh{display:flex;justify-content:space-between;font-size:0.74rem;margin-bottom:4px;}
.sn2{font-weight:700;}.sp2{color:var(--muted);font-weight:400;}
.st{height:6px;background:rgba(255,255,255,0.07);border-radius:7px;overflow:hidden;}
.sf{height:100%;border-radius:7px;transition:width 1.2s ease;}
.fill-buy{background:linear-gradient(90deg,var(--green),var(--teal));}
.fill-wait{background:linear-gradient(90deg,var(--yellow),var(--gold));}
.fill-avoid{background:linear-gradient(90deg,var(--red),#ff6b35);}
.sl2{display:flex;justify-content:space-between;font-size:0.6rem;margin-top:3px;opacity:0.65;}
/* Orders */
.os{background:rgba(0,0,0,0.2);border:1px solid var(--border);border-radius:11px;padding:11px;margin-bottom:9px;}
.os-sl{border-color:rgba(255,61,87,0.25);background:rgba(255,61,87,0.03);}
.os-trail{border-color:rgba(0,188,212,0.25);background:rgba(0,188,212,0.03);}
.ot{font-size:0.67rem;color:var(--muted);margin-bottom:9px;font-weight:600;letter-spacing:0.4px;}
.og1{display:grid;grid-template-columns:1fr;gap:6px;}
.og2{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.og3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;}
.oi{text-align:center;}
.ol{font-size:0.61rem;color:var(--muted);margin-bottom:3px;}
.ov{font-size:0.95rem;font-weight:700;padding:3px 5px;border-radius:6px;display:inline-block;}
.ov.entry{color:var(--text)}.ov.tp{color:var(--green)}.ov.sl{color:var(--red)}.ov.rr{color:var(--gold)}
.oh{font-size:0.58rem;color:var(--muted);margin-top:2px;}
.cv{cursor:pointer;transition:all 0.15s;border-radius:5px;padding:2px 4px;}
.cv:hover{background:rgba(255,255,255,0.08);}
.cv:active{transform:scale(0.93);}
.copied{background:rgba(0,230,118,0.15)!important;color:var(--green)!important;}
.rr-row{display:flex;justify-content:space-between;font-size:0.7rem;color:var(--muted);margin-top:8px;padding-top:7px;border-top:1px solid var(--border);}
.trail-note{font-size:0.63rem;color:var(--teal);margin-top:7px;padding-top:7px;border-top:1px solid rgba(0,188,212,0.2);}
.pos{color:var(--green)}.neg{color:var(--red)}
.stars{color:var(--gold);font-size:0.83rem;cursor:help;}
.exp-badge{background:rgba(255,179,0,0.12);color:var(--yellow);border:1px solid rgba(255,179,0,0.3);padding:2px 6px;border-radius:6px;font-size:0.6rem;font-weight:600;}
.cust-badge{background:rgba(41,121,255,0.12);color:var(--blue);border:1px solid rgba(41,121,255,0.3);padding:2px 6px;border-radius:6px;font-size:0.6rem;}
.cb{background:none;border:1px solid var(--border);color:var(--muted);padding:5px 11px;border-radius:7px;font-family:inherit;font-size:0.68rem;cursor:pointer;width:100%;margin-top:4px;}
.cleg{font-size:0.6rem;color:var(--muted);margin-top:7px;opacity:0.65;}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:3px;margin-top:7px;}
.ci{display:flex;align-items:center;gap:4px;font-size:0.64rem;padding:3px 4px;border-radius:5px;background:rgba(255,255,255,0.025);}
.co{color:var(--green)}.cf{color:var(--red);opacity:0.55;}
.cd{width:5px;height:5px;border-radius:50%;flex-shrink:0;}
.dok{background:var(--green)}.dfail{background:var(--red);opacity:0.45;}
.cw{color:var(--gold);font-size:0.58rem;margin-right:auto;}
.btn-row{display:flex;gap:6px;margin-top:8px;}
.del-btn{flex:1;background:rgba(255,61,87,0.07);border:1px solid rgba(255,61,87,0.25);color:var(--red);padding:5px;border-radius:7px;font-family:inherit;font-size:0.68rem;cursor:pointer;transition:all 0.2s;}
.del-btn:hover{background:rgba(255,61,87,0.18);}
.excl-btn{flex:1;background:rgba(121,134,203,0.07);border:1px solid rgba(121,134,203,0.25);color:var(--muted);padding:5px;border-radius:7px;font-family:inherit;font-size:0.68rem;cursor:pointer;transition:all 0.2s;}
.excl-btn:hover{background:rgba(121,134,203,0.18);color:var(--text);}
/* Custom panel */
.cp{background:var(--card);border:1px solid var(--border);border-radius:13px;padding:13px;margin-bottom:16px;}
.cp h3{font-size:0.85rem;color:var(--teal);margin-bottom:9px;}
.ci-row{display:flex;gap:6px;flex-wrap:wrap;}
.ci-inp{background:rgba(0,0,0,0.3);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:8px;font-family:inherit;font-size:0.8rem;flex:1;min-width:85px;}
.ci-inp:focus{outline:none;border-color:var(--blue);}
.add-btn{background:rgba(0,230,118,0.1);border:1px solid var(--green);color:var(--green);padding:7px 13px;border-radius:8px;cursor:pointer;font-family:inherit;font-size:0.8rem;}
.ecl{display:inline-block;background:rgba(255,61,87,0.07);border:1px solid rgba(255,61,87,0.22);color:var(--red);padding:3px 9px;border-radius:13px;font-size:0.68rem;cursor:pointer;margin:3px;}
.ecl:hover{background:rgba(255,61,87,0.18);}
/* Legend */
.lgd{background:rgba(0,0,0,0.18);border:1px solid var(--border);border-radius:10px;padding:11px;margin-bottom:16px;font-size:0.72rem;color:var(--muted);}
.lgd h4{color:var(--teal);margin-bottom:7px;font-size:0.78rem;}
.lgd-g{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:5px;}
/* Loading */
.lo{position:fixed;inset:0;background:rgba(10,14,26,0.93);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:100;}
.lo.show{display:flex;}
.lr{width:62px;height:62px;border:3px solid var(--border);border-top-color:var(--teal);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:13px;}
@keyframes spin{to{transform:rotate(360deg)}}
.lt{color:var(--teal);font-size:0.92rem;}.ls2{color:var(--muted);font-size:0.72rem;margin-top:5px;}
.nr{text-align:center;padding:48px 20px;color:var(--muted);}
.toast{position:fixed;bottom:22px;left:50%;transform:translateX(-50%);background:var(--green);color:#000;padding:7px 18px;border-radius:18px;font-size:0.78rem;font-weight:600;opacity:0;transition:opacity 0.3s;z-index:200;pointer-events:none;}
.toast.show{opacity:1;}
@media(max-width:768px){
  .wrapper{padding:12px;}
  .header{padding:18px 12px 12px;}
  .header h1{font-size:1.6rem;letter-spacing:1px;}
  .tabs{gap:5px;margin-bottom:12px;}
  .tab-btn{padding:8px 14px;font-size:0.82rem;border-radius:40px;}
  .cards-grid{grid-template-columns:1fr;gap:12px;}
  .sc{padding:14px;border-radius:14px;}
  .sn{font-size:0.98rem;}
  .sp{font-size:1.2rem;}
  .og1,.og2,.og3{grid-template-columns:1fr 1fr;gap:8px;}
  .ov{font-size:1rem;padding:5px 6px;}
  .ol{font-size:0.65rem;}
  .oh{font-size:0.62rem;}
  .os{padding:10px;}
  .ot{font-size:0.7rem;}
  .stats-bar{gap:7px;}
  .stat-card{min-width:70px;padding:9px 10px;}
  .stat-card .num{font-size:1.45rem;}
  .stat-card .lbl{font-size:0.62rem;}
  .scan-btn{padding:12px 22px;font-size:1rem;width:100%;justify-content:center;margin-top:10px;}
  .cp{padding:11px;}
  .ci-row{gap:5px;}
  .ci-inp{font-size:0.82rem;padding:8px 9px;}
  .add-btn{width:100%;padding:9px;font-size:0.85rem;margin-top:4px;}
  .btn-row{gap:5px;}
  .del-btn,.excl-btn{padding:7px;font-size:0.72rem;}
  .exp-grid{grid-template-columns:1fr;}
  .exp-mini-card{padding:9px 12px;}
  .lgd{padding:10px;font-size:0.7rem;}
  .lgd-g{grid-template-columns:1fr;}
  .rr-row{font-size:0.68rem;}
  .trail-note{font-size:0.65rem;}
  .tr{gap:4px;}
  .tc{padding:3px 7px;font-size:0.68rem;}
  .vb{font-size:0.76rem;padding:4px 12px;}
  .cb{padding:7px;font-size:0.72rem;}
  .cg{grid-template-columns:1fr;}
}
@media(max-width:380px){
  .tab-btn{padding:6px 10px;font-size:0.75rem;}
  .header h1{font-size:1.35rem;}
  .og2{grid-template-columns:1fr;}
}
</style>"""

JS = """<script>
var activeTab='tadawul';
function switchTab(t){
  activeTab=t;
  document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c=>c.classList.remove('active'));
  document.getElementById('tab-'+t).classList.add('active');
  document.getElementById('content-'+t).classList.add('active');
}
function startScan(){
  document.getElementById('lo').classList.add('show');
  var lb={'tadawul':'السوق السعودي','us':'السوق الأمريكي','crypto':'العملات'};
  document.getElementById('lm').textContent=lb[activeTab]||activeTab;
  fetch('/scan?market='+activeTab).then(r=>r.json()).then(()=>checkStatus());
}
function checkStatus(){
  fetch('/status?market='+activeTab).then(r=>r.json()).then(d=>{
    if(d.status==='done') window.location.reload();
    else setTimeout(checkStatus,2000);
  });
}
function copyVal(el){
  var t=el.textContent.trim();
  navigator.clipboard.writeText(t).then(()=>{
    el.classList.add('copied');
    showToast('تم النسخ: '+t);
    setTimeout(()=>el.classList.remove('copied'),1400);
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
  var market=document.getElementById('nMkt').value;
  if(!code||!name){showToast('أدخل الرمز والاسم');return;}
  fetch('/add_stock',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,name,market})})
  .then(r=>r.json()).then(d=>{if(d.ok){showToast('تمت الإضافة ✓');setTimeout(()=>window.location.reload(),800);}});
}
function deleteStock(code,market){
  if(!confirm('حذف '+code+'؟')) return;
  fetch('/delete_stock',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code,market})})
  .then(r=>r.json()).then(d=>{if(d.ok) window.location.reload();});
}
function excludeStock(code){
  fetch('/exclude',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code})})
  .then(r=>r.json()).then(d=>{if(d.ok) window.location.reload();});
}
function removeExclusion(code){
  fetch('/include',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code})})
  .then(r=>r.json()).then(d=>{if(d.ok) window.location.reload();});
}
</script>"""

@app.route("/")
def index():
    custom=load_custom(); excluded=custom.get("excluded",[])
    html=(
        '<!DOCTYPE html><html lang="ar" dir="rtl"><head>'
        '<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">'
        '<title>جلال سكانر</title>'
        '<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600;700&display=swap" rel="stylesheet">'
        +CSS+'</head><body>'
        '<div id="toast" class="toast"></div>'
        '<div class="lo" id="lo"><div class="lr"></div>'
        '<div class="lt">🔍 جاري مسح <span id="lm"></span>...</div>'
        '<div class="ls2">قد يأخذ 1-3 دقائق</div></div>'
        '<div class="wrapper">'
        '<div class="header"><h1>⚡ جلال سكانر</h1>'
        '<div class="sub">Jalal Scanner v2.2 — تاسي + أمريكي + عملات</div></div>'
        '<div class="tabs">'
        '<button class="tab-btn active" id="tab-tadawul" onclick="switchTab(\'tadawul\')">🇸🇦 السوق السعودي</button>'
        '<button class="tab-btn" id="tab-us" onclick="switchTab(\'us\')">🇺🇸 السوق الأمريكي</button>'
        '<button class="tab-btn" id="tab-crypto" onclick="switchTab(\'crypto\')">💰 العملات الرقمية</button>'
        '</div>'
    )

    # Legend
    html+=(
        '<div class="lgd"><h4>📖 دليل سريع</h4><div class="lgd-g">'
        '<div>⭐⭐⭐ استثمار — فوق الثلاثة — هدف أشهر</div>'
        '<div>⭐⭐ سوينج — يومي+أسبوعي — هدف أسابيع</div>'
        '<div>⭐ مضاربة — يومي فقط — هدف أيام</div>'
        '<div>⏳ انتظر — إشارة ناقصة — لا تدخل</div>'
        '<div>❌ تجنب — تحت المتوسطات — ابتعد</div>'
        '<div>JRF 15+/20 قوي | 10-14 متوسط | أقل ضعيف</div>'
        '<div>🚀 انفجار = حجم تداول شاذ × 3 أو أكثر</div>'
        '<div>اضغط أي رقم في الأوامر لنسخه فوراً</div>'
        '</div></div>'
    )

    # Custom panel
    html+=(
        '<div class="cp"><h3>➕ إضافة / حذف أسهم</h3>'
        '<div class="ci-row">'
        '<input class="ci-inp" id="nCode" placeholder="الرمز" style="max-width:120px;">'
        '<input class="ci-inp" id="nName" placeholder="الاسم">'
        '<select class="ci-inp" id="nMkt" style="max-width:130px;">'
        '<option value="tadawul">🇸🇦 تاسي</option>'
        '<option value="us">🇺🇸 أمريكي</option>'
        '<option value="crypto">💰 عملات</option>'
        '</select>'
        '<button class="add-btn" onclick="addStock()">➕ إضافة</button>'
        '</div>'
    )
    if excluded:
        html+='<div style="margin-top:9px;font-size:0.7rem;color:var(--muted);">مستبعدون — اضغط للإعادة:</div><div>'
        for code in excluded:
            html+='<span class="ecl" onclick="removeExclusion(\''+code+'\')">'+code+' ↩</span>'
        html+='</div>'
    html+='</div>'

    # Tabs
    for market,label,flag in [("tadawul","السوق السعودي","🇸🇦"),("us","السوق الأمريكي","🇺🇸"),("crypto","العملات","💰")]:
        data=scan_state[market]["data"] or []
        last=scan_state[market]["last_scan"] or ""
        active_cls=" active" if market=="tadawul" else ""
        html+='<div class="tab-content'+active_cls+'" id="content-'+market+'">'
        html+='<div style="text-align:center;padding:7px 0 13px;">'
        html+='<div class="ls">'+('آخر مسح: '+last if last else 'لم يتم المسح بعد')+'</div>'
        html+='<button class="scan-btn" onclick="startScan()">🔍 مسح '+flag+' '+label+'</button>'
        html+='</div>'

        if data:
            buy_list=[s for s in data if s['verdict']=='BUY']
            wait_list=[s for s in data if s['verdict']=='WAIT']
            avoid_list=[s for s in data if s['verdict']=='AVOID']
            strong=[s for s in data if s['stars']>=2]
            exp_list=[s for s in data if s.get('explosion')]

            html+=(
                '<div class="stats-bar">'
                '<div class="stat-card gold"><div class="num">'+str(len(data))+'</div><div class="lbl">إجمالي</div></div>'
                '<div class="stat-card green"><div class="num">'+str(len(buy_list))+'</div><div class="lbl">🟢 BUY</div></div>'
                '<div class="stat-card yellow"><div class="num">'+str(len(wait_list))+'</div><div class="lbl">🟡 WAIT</div></div>'
                '<div class="stat-card red"><div class="num">'+str(len(avoid_list))+'</div><div class="lbl">🔴 AVOID</div></div>'
                '<div class="stat-card"><div class="num" style="color:var(--teal)">'+str(len(strong))+'</div><div class="lbl">⭐⭐+ قوي</div></div>'
                '<div class="stat-card" style="border-color:var(--gold)"><div class="num" style="color:var(--gold)">'+str(len(exp_list))+'</div><div class="lbl">🚀 انفجار</div></div>'
                '</div>'
            )

            # قسم الانفجارات
            if exp_list:
                html+='<div class="exp-section"><div class="exp-section-title">🚀 تنبيهات الانفجار — حجم تداول شاذ</div><div class="exp-grid">'
                for s in exp_list:
                    sfx=".SR" if s['market']=="tadawul" else ""
                    vl=s['verdict'].lower()
                    var_short="🟢 اشتري" if s['verdict']=='BUY' else ("🟡 انتظر" if s['verdict']=='WAIT' else "🔴 تجنب")
                    html+=(
                        '<div class="exp-mini-card">'
                        '<div><div class="exp-mini-name">'+s['name']+'</div>'
                        '<div class="exp-mini-code">'+s['code']+sfx+'</div></div>'
                        '<div class="exp-mini-price">'+str(s['price'])+'</div>'
                        '<div class="exp-mini-ratio">🚀 ×'+str(s['vol_ratio'])+'</div>'
                        '<div class="exp-mini-verdict">'+var_short+'</div>'
                        '</div>'
                    )
                html+='</div></div>'

            if buy_list:
                html+='<div class="sec-t sec-buy">🎯 جاهز للدخول — BUY</div><div class="cards-grid">'
                for i,s in enumerate(buy_list): html+=render_card(s,i)
                html+='</div>'
            if wait_list:
                html+='<div class="sec-t sec-wait">👀 تحت المراقبة — WAIT</div><div class="cards-grid">'
                for i,s in enumerate(wait_list): html+=render_card(s,i)
                html+='</div>'
            if avoid_list:
                html+='<div class="sec-t sec-avoid">⛔ تجنب — AVOID</div><div class="cards-grid">'
                for i,s in enumerate(avoid_list): html+=render_card(s,i)
                html+='</div>'
        else:
            html+='<div class="nr"><div style="font-size:2.3rem">📡</div><div style="margin-top:7px">اضغط "مسح" لتحليل '+label+'</div></div>'
        html+='</div>'

    html+='</div>'+JS+'</body></html>'
    return html

@app.route("/scan")
def scan():
    market=request.args.get("market","tadawul")
    if scan_state[market]["status"]=="scanning":
        return jsonify({"status":"already_running"})
    t=threading.Thread(target=run_scan,args=(market,))
    t.daemon=True; t.start()
    return jsonify({"status":"started"})

@app.route("/status")
def status():
    market=request.args.get("market","tadawul")
    return jsonify({"status":scan_state[market]["status"]})

@app.route("/add_stock",methods=["POST"])
def add_stock():
    data=request.get_json()
    code=data.get("code","").strip(); name=data.get("name","").strip(); market=data.get("market","tadawul")
    if not code or not name: return jsonify({"ok":False})
    custom=load_custom()
    custom.setdefault(market,{})[code]=name
    if code in custom.get("excluded",[]): custom["excluded"].remove(code)
    save_custom(custom); return jsonify({"ok":True})

@app.route("/delete_stock",methods=["POST"])
def delete_stock():
    data=request.get_json()
    code=data.get("code","").strip(); market=data.get("market","tadawul")
    custom=load_custom()
    if code in custom.get(market,{}): del custom[market][code]
    save_custom(custom); return jsonify({"ok":True})

@app.route("/exclude",methods=["POST"])
def exclude():
    code=request.get_json().get("code","")
    custom=load_custom()
    if code not in custom.get("excluded",[]): custom.setdefault("excluded",[]).append(code)
    save_custom(custom); return jsonify({"ok":True})

@app.route("/include",methods=["POST"])
def include():
    code=request.get_json().get("code","")
    custom=load_custom()
    if code in custom.get("excluded",[]): custom["excluded"].remove(code)
    save_custom(custom); return jsonify({"ok":True})

if __name__=="__main__":
    print("="*55)
    print("  ⚡ جلال سكانر v2.2")
    print("  http://localhost:5000")
    print("  من الجوال: http://192.168.0.113:5000")
    print("="*55)
    threading.Timer(1.5,lambda:webbrowser.open("http://localhost:5000")).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
