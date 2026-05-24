#!/usr/bin/env python3
# ══════════════════════════════════════════════════
# جلال رادار v5.0 — Clean & Smart Edition
# ══════════════════════════════════════════════════
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings, threading, webbrowser, os, json, random, time
warnings.filterwarnings("ignore")

app = Flask(__name__)

CUSTOM_FILE = os.path.join(os.getcwd(), "custom_stocks.json")
USD_TO_SAR = 3.75

def load_custom():
    if os.path.exists(CUSTOM_FILE):
        with open(CUSTOM_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"tadawul": {}, "us": {}, "crypto": {}, "excluded": []}

def save_custom(d):
    with open(CUSTOM_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

# ══ قوائم الأسهم ══
DEFAULT_TADAWUL = {
    "2222":"أرامكو","7010":"STC","1120":"الراجحي","1010":"الرياض","1080":"الأهلي",
    "1050":"الإنماء","2010":"سابك","2020":"سابك المغذيات","4190":"جرير",
    "4003":"إكسترا","4001":"سينومي ريتيل","4321":"سينومي سنترز","2082":"أكوا باور",
    "4072":"MBC","7030":"زين","7020":"إتحاد إتصالات","7040":"قو للإتصالات",
    "5110":"الكهرباء","4013":"سليمان الحبيب","4007":"الحمادي","4009":"السعودي الألماني",
    "4015":"جمجوم فارما","4164":"النهدي","4163":"الدواء","2280":"المراعي",
    "6002":"هرفي","6004":"كاتريون","6016":"برغرايززر","2050":"صافولا",
    "4250":"جبل عمر","4300":"دار الأركان","4020":"العقارية","4090":"طيبة",
    "1211":"معادن","2290":"ينساب","2350":"كيان","2060":"التصنيع",
    "4030":"البحري","4040":"سابتكو","4050":"ساسكو","4263":"سال",
    "1183":"سهل","7202":"سلوشنز","1111":"تداول","4084":"دراية",
    "4322":"رتال","4232":"مدى","4325":"مسار","3030":"أسمنت السعودية",
    "3020":"أسمنت اليمامة","3090":"أسمنت تبوك","3060":"أسمنت ينبع",
    "2083":"مرافق","2082":"أكوا","4017":"فقيه الطبية","4018":"الموسى",
    "4002":"المواساة","2330":"المتقدمة","1302":"بوان","2310":"سبكيم",
    "4160":"ثمار","4130":"درب السعودية","2281":"تنمية","2282":"نقي",
    "4291":"الوطنية للتعليم","4290":"الخليج للتدريب","1831":"مهارة",
    "6040":"تبوك الزراعية","2285":"المطاحن العربية","2286":"المطاحن الرابعة",
    "4144":"رؤوم","4005":"رعاية","2223":"لوبريف","4031":"الخدمات الأرضية",
    "2370":"مسك","4180":"فتيحي","4011":"لازوردي","4192":"السيف غاليري",
    "2240":"صناعات","2300":"صناعة الورق","1320":"أنابيب السعودية",
    "2110":"الكابلات","2160":"أميانتيت","1304":"اليمامة للحديد",
    "3050":"أسمنت الجنوب","3080":"أسمنت الشرقية","3092":"أسمنت الرياض",
    "3010":"أسمنت العربية","3005":"أسمنت ام القرى","3003":"أسمنت المدينة",
    "4080":"سناد","2140":"أيان","2120":"متطورة","2190":"سيسكو",
    "1810":"سيرا","4194":"محطة البناء","4162":"المنجم","2381":"الحفر العربية",
}

DEFAULT_US = {
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet","META":"Meta",
    "NVDA":"NVIDIA","AMD":"AMD","TSLA":"Tesla","AMZN":"Amazon",
    "INTC":"Intel","QCOM":"Qualcomm","AVGO":"Broadcom","TXN":"Texas Instruments",
    "MU":"Micron","AMAT":"Applied Materials","LRCX":"Lam Research","KLAC":"KLA",
    "CRM":"Salesforce","NOW":"ServiceNow","SNOW":"Snowflake","DDOG":"Datadog",
    "HUBS":"HubSpot","WDAY":"Workday","ADSK":"Autodesk","ORCL":"Oracle",
    "INTU":"Intuit","CDNS":"Cadence","SNPS":"Synopsys","VEEV":"Veeva",
    "ZS":"Zscaler","CRWD":"CrowdStrike","PANW":"Palo Alto","FTNT":"Fortinet",
    "NET":"Cloudflare","OKTA":"Okta","CYBR":"CyberArk","S":"SentinelOne",
    "JNJ":"Johnson & Johnson","UNH":"UnitedHealth","ABBV":"AbbVie",
    "TMO":"Thermo Fisher","ABT":"Abbott","DHR":"Danaher","ISRG":"Intuitive Surgical",
    "SYK":"Stryker","BSX":"Boston Scientific","VRTX":"Vertex","REGN":"Regeneron",
    "COST":"Costco","HD":"Home Depot","WMT":"Walmart","TGT":"Target",
    "NKE":"Nike","SBUX":"Starbucks","MCD":"McDonald's","CMG":"Chipotle",
    "LULU":"Lululemon","ULTA":"Ulta Beauty","ROST":"Ross","TJX":"TJX",
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","EOG":"EOG",
    "OXY":"Occidental","SLB":"Schlumberger","HAL":"Halliburton",
    "FSLR":"First Solar","NEE":"NextEra","ENPH":"Enphase","SEDG":"SolarEdge",
    "LIN":"Linde","APD":"Air Products","ECL":"Ecolab","SHW":"Sherwin-Williams",
    "ALB":"Albemarle","HON":"Honeywell","CAT":"Caterpillar","DE":"John Deere",
    "EMR":"Emerson","ETN":"Eaton","ROK":"Rockwell","GE":"GE Aerospace",
    "UBER":"Uber","LYFT":"Lyft","ABNB":"Airbnb","BKNG":"Booking","EXPE":"Expedia",
    "UPS":"UPS","FDX":"FedEx","ODFL":"Old Dominion","JBHT":"JB Hunt",
    "PYPL":"PayPal","SQ":"Block","AFRM":"Affirm","SOFI":"SoFi","BILL":"Bill.com",
    "NFLX":"Netflix","SPOT":"Spotify","RBLX":"Roblox","EA":"EA","TTWO":"Take-Two",
    "LEN":"Lennar","DHI":"D.R. Horton","PHM":"PulteGroup","TOL":"Toll Brothers",
    "WM":"Waste Management","RSG":"Republic Services","CLH":"Clean Harbors",
    "DUOL":"Duolingo","COUR":"Coursera","LOPE":"Grand Canyon",
    "TSLA":"Tesla","PATH":"UiPath","AI":"C3.ai","APP":"AppLovin","ASAN":"Asana",
    "ADM":"ADM","BG":"Bunge","TSN":"Tyson","GIS":"General Mills","HSY":"Hershey",
    "GEHC":"GE HealthCare","DXCM":"DexCom","ALGN":"Align","ISRG":"Intuitive",
    "NUE":"Nucor","STLD":"Steel Dynamics","RS":"Reliance Steel","CMC":"Commercial Metals",
    "MRVL":"Marvell","ON":"ON Semi","MPWR":"Monolithic Power","ADI":"Analog Devices",
}

DEFAULT_CRYPTO = {
    "BTC-USD":"Bitcoin","ETH-USD":"Ethereum","SOL-USD":"Solana",
    "BNB-USD":"BNB","XRP-USD":"Ripple","ADA-USD":"Cardano",
    "AVAX-USD":"Avalanche","DOT-USD":"Polkadot","LINK-USD":"Chainlink",
    "MATIC-USD":"Polygon","ATOM-USD":"Cosmos","UNI-USD":"Uniswap",
    "LTC-USD":"Litecoin","NEAR-USD":"NEAR","ARB-USD":"Arbitrum",
    "OP-USD":"Optimism","INJ-USD":"Injective","SUI-USD":"Sui",
    "APT-USD":"Aptos","FET-USD":"Fetch.AI","RENDER-USD":"Render",
    "DOGE-USD":"Dogecoin","SHIB-USD":"Shiba Inu","PEPE-USD":"Pepe",
    "TRX-USD":"Tron","TON-USD":"Toncoin","KAS-USD":"Kaspa",
    "AAVE-USD":"Aave","SAND-USD":"Sandbox","MANA-USD":"Decentraland",
}

# ══ المفضلة ══
FAVORITES_FILE = os.path.join(os.getcwd(), "favorites.json")

def load_favorites():
    if os.path.exists(FAVORITES_FILE):
        with open(FAVORITES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}  # {"tadawul": ["2222","7010"], "us": ["AAPL"], "crypto": []}

def save_favorites(d):
    with open(FAVORITES_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

BENCHMARK = {"tadawul": "^TASI.SR", "us": "^GSPC", "crypto": "BTC-USD"}

scan_state = {
    "tadawul": {"data": None, "last_scan": None, "status": "idle", "progress": 0, "total": 0, "period": "daily"},
    "us":      {"data": None, "last_scan": None, "status": "idle", "progress": 0, "total": 0, "period": "daily"},
    "crypto":  {"data": None, "last_scan": None, "status": "idle", "progress": 0, "total": 0, "period": "daily"},
}

# ══ دوال التحليل ══
def get_df(t, p, i):
    try:
        df = yf.download(t, period=p, interval=i, progress=False, auto_adjust=True, timeout=10)
        if df.empty: return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df
    except: return pd.DataFrame()

def ema(s, p): return s.ewm(span=p, adjust=False).mean()
def sma(s, p): return s.rolling(p).mean()

def rsi_f(s, p=14):
    d = s.diff(); g = d.clip(lower=0).rolling(p).mean()
    l = (-d.clip(upper=0)).rolling(p).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def macd_f(s):
    ml = ema(s, 12) - ema(s, 26); sg = ema(ml, 9); return ml, sg, ml - sg

def adx_f(h, l, c, p=14):
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    at = tr.rolling(p).mean()
    up, dn = h.diff(), -l.diff()
    dp = pd.Series(np.where((up > dn) & (up > 0), up, 0.), index=h.index).rolling(p).mean()
    dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.), index=h.index).rolling(p).mean()
    dip, dim = 100*dp/at, 100*dm/at
    return (100*(dip-dim).abs()/(dip+dim).replace(0, np.nan)).rolling(p).mean()

def atr_f(h, l, c, p=14):
    return pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1).rolling(p).mean()

def stoch_f(h, l, c, k=14, d=3):
    kv = 100*(c-l.rolling(k).min())/(h.rolling(k).max()-l.rolling(k).min()).replace(0, np.nan)
    return kv, kv.rolling(d).mean()

def analyze(code, name, market, bdf=None, period="daily"):
    try:
        ticker = code + ".SR" if market == "tadawul" else code
        # نختار الإطار الزمني حسب الفترة
        if period == "weekly":
            dfd = get_df(ticker, "5y", "1wk")
            dfw = dfd
            dfm = get_df(ticker, "10y", "1mo")
        elif period == "monthly":
            dfd = get_df(ticker, "10y", "1mo")
            dfw = dfd
            dfm = dfd
        else:  # daily
            dfd = get_df(ticker, "2y", "1d")
            dfw = get_df(ticker, "5y", "1wk")
            dfm = get_df(ticker, "10y", "1mo")

        if dfd.empty or len(dfd) < 30: return None

        price = float(dfd["Close"].iloc[-1])

        # ══ فلترات الجودة ══
        # 1. فلتر السعر
        if market == "us" and price < 5.0: return None
        if market == "tadawul" and price < 1.0: return None

        # 2. فلتر السيولة اليومية
        try:
            avg_vol = float(sma(dfd["Volume"], 20).iloc[-1])
            daily_liquidity = price * avg_vol
            if market == "us" and daily_liquidity < 5_000_000: return None      # أقل من 5 مليون دولار = استبعاد
            if market == "tadawul" and daily_liquidity < 10_000_000: return None  # أقل من 10 مليون ريال = استبعاد
        except: pass

        # 3. فلتر Market Cap عبر yfinance
        try:
            tk = yf.Ticker(ticker)
            info = tk.fast_info
            mkt_cap = getattr(info, 'market_cap', None)
            if mkt_cap:
                if market == "us" and mkt_cap < 500_000_000: return None       # أقل من 500 مليون دولار = استبعاد
                if market == "tadawul" and mkt_cap < 500_000_000: return None   # أقل من 500 مليون ريال = استبعاد
        except: pass

        # السعر بالدولار والريال للسوق الأمريكي
        price_usd = price if market == "us" else None
        price_sar = round(price * USD_TO_SAR, 2) if market == "us" else None

        e20d = float(ema(dfd["Close"], 20).iloc[-1])
        e20w = float(ema(dfw["Close"], 20).iloc[-1]) if len(dfw) >= 20 else None
        e20m = float(ema(dfm["Close"], 20).iloc[-1]) if len(dfm) >= 20 else None
        ad = price > e20d
        aw = price > e20w if e20w else False
        am = price > e20m if e20m else False

        if ad and aw and am: trend, stars = "استثمار", 3
        elif ad and aw: trend, stars = "سوينج", 2
        elif ad: trend, stars = "مضاربة", 1
        else: trend, stars = "تجنب", 0

        d = dfd.copy()
        e20 = ema(d["Close"], 20); e50 = ema(d["Close"], 50); e200 = ema(d["Close"], 200)
        rv = rsi_f(d["Close"]); ml, sg, mh = macd_f(d["Close"])
        adv = adx_f(d["High"], d["Low"], d["Close"])
        sk, sdv = stoch_f(d["High"], d["Low"], d["Close"])
        va = sma(d["Volume"], 20)
        av = atr_f(d["High"], d["Low"], d["Close"])

        c1 = price > float(e20.iloc[-1]); c2 = float(e20.iloc[-1]) > float(e50.iloc[-1])
        c3 = price > float(e200.iloc[-1]); c4 = float(ml.iloc[-1]) > float(sg.iloc[-1])
        c5 = float(mh.iloc[-1]) > float(mh.iloc[-2]); c6 = 40 <= float(rv.iloc[-1]) <= 70
        c7 = float(rv.iloc[-1]) > float(rv.iloc[-2]); c8 = float(adv.iloc[-1]) > 20
        c9 = float(sk.iloc[-1]) > 20 and float(sk.iloc[-1]) > float(sdv.iloc[-1])
        c10 = float(d["Volume"].iloc[-1]) > float(va.iloc[-1]) * 1.2
        c11 = price > float(e20.iloc[-1])
        c12 = float(adv.iloc[-1]) > float(adv.iloc[-2])

        score = (2 if c1 else 0)+(2 if c2 else 0)+(2 if c3 else 0)+(2 if c4 else 0)+\
                (1 if c5 else 0)+(2 if c6 else 0)+(1 if c7 else 0)+(2 if c8 else 0)+\
                (2 if c9 else 0)+(2 if c10 else 0)+(1 if c11 else 0)+(1 if c12 else 0)

        atr_v = float(av.iloc[-1]); p = round(price, 3)
        lb = round(p * 0.995, 3)
        t1 = round(p + atr_v * 2.0, 3); t2 = round(p + atr_v * 4.0, 3)
        sl = round(p - atr_v * 1.5, 3)
        rr = round((t1 - p) / max(p - sl, 0.001), 2)
        psl = round((p - sl) / p * 100, 2)
        ptp = round((t1 - p) / p * 100, 2)
        liq = round(price * float(va.iloc[-1]) / 1e6, 1) if market == "tadawul" else round(price * float(va.iloc[-1]) / 1e6, 1)
        liq_unit = "م.ر" if market == "tadawul" else "م.$"

        vr = float(d["Volume"].iloc[-1]) / float(va.iloc[-1]) if float(va.iloc[-1]) > 0 else 1
        rsi_val = round(float(rv.iloc[-1]), 1)
        adx_val = round(float(adv.iloc[-1]), 1)

        if score >= 15 and rr >= 1.3: verdict, bt = "BUY", "شراء"
        elif score >= 13 and rr >= 1.0 and ad and aw: verdict, bt = "BUY", "شراء مشروط"
        elif score >= 10: verdict, bt = "WAIT", "انتظر"
        else: verdict, bt = "AVOID", "تجنب"

        # نسبة الثقة
        confidence = min(100, round((score / 20) * 60 + (min(rr, 3) / 3) * 20 + (stars / 3) * 20))

        # تغير السعر
        prev = float(dfd["Close"].iloc[-2]) if len(dfd) > 1 else price
        chg = round((price - prev) / prev * 100, 2)

        return {
            "code": code, "name": name, "price": p, "market": market,
            "price_usd": price_usd, "price_sar": price_sar,
            "stars": stars, "trend": trend,
            "above_daily": ad, "above_weekly": aw, "above_monthly": am,
            "score": score, "verdict": verdict, "bt": bt,
            "lb": lb, "t1": t1, "t2": t2, "sl": sl,
            "rr": rr, "psl": psl, "ptp": ptp,
            "rsi": rsi_val, "adx": adx_val, "vr": round(vr, 1),
            "liq": liq, "liq_unit": liq_unit,
            "confidence": confidence, "chg": chg,
            "is_custom": False, "rank_score": 0,
        }
    except: return None

def analyze_crypto(code, name, bdf=None, period="daily"):
    try:
        if period == "weekly":
            dfd = get_df(code, "5y", "1wk")
        elif period == "monthly":
            dfd = get_df(code, "10y", "1mo")
        else:
            dfd = get_df(code, "2y", "1d")

        if dfd.empty or len(dfd) < 30: return None

        price = float(dfd["Close"].iloc[-1])

        # ══ فلترات جودة الكريبتو ══
        if price < 0.000001: return None

        # فلتر السيولة — استبعاد العملات ذات التداول المنخفض
        try:
            avg_vol = float(sma(dfd["Volume"], 20).iloc[-1])
            daily_liq_usd = price * avg_vol
            if daily_liq_usd < 1_000_000: return None  # أقل من مليون دولار يومياً = استبعاد
        except: pass

        price_usd = round(price, 4)
        price_sar = round(price * USD_TO_SAR, 2)

        e20d = float(ema(dfd["Close"], 20).iloc[-1])
        dfw = get_df(code, "5y", "1wk")
        e20w = float(ema(dfw["Close"], 20).iloc[-1]) if len(dfw) >= 20 else None
        ad = price > e20d; aw = price > e20w if e20w else False

        if ad and aw: trend, stars = "سوينج", 2
        elif ad: trend, stars = "مضاربة", 1
        else: trend, stars = "تجنب", 0

        d = dfd.copy()
        rv = rsi_f(d["Close"]); adv = adx_f(d["High"], d["Low"], d["Close"])
        av = atr_f(d["High"], d["Low"], d["Close"])
        ml, sg, mh = macd_f(d["Close"])

        atr_v = float(av.iloc[-1]); p = round(price, 4)
        t1 = round(p + atr_v * 2.0, 4); t2 = round(p + atr_v * 4.0, 4)
        sl = round(p - atr_v * 1.5, 4)
        rr = round((t1 - p) / max(p - sl, 0.001), 2)
        psl = round((p - sl) / p * 100, 2)
        ptp = round((t1 - p) / p * 100, 2)
        rsi_val = round(float(rv.iloc[-1]), 1)
        adx_val = round(float(adv.iloc[-1]), 1)

        c4 = float(ml.iloc[-1]) > float(sg.iloc[-1])
        c6 = 40 <= rsi_val <= 70; c8 = adx_val > 20
        score = (4 if (ad and aw) else (2 if ad else 0)) + \
                (3 if c4 else 0) + (2 if c6 else 0) + (2 if c8 else 0)

        if score >= 8 and rr >= 1.3: verdict, bt = "BUY", "شراء"
        elif score >= 6 and ad: verdict, bt = "BUY", "شراء مشروط"
        elif score >= 4: verdict, bt = "WAIT", "انتظر"
        else: verdict, bt = "AVOID", "تجنب"

        confidence = min(100, round((score / 11) * 70 + (min(rr, 3) / 3) * 30))

        prev = float(dfd["Close"].iloc[-2]) if len(dfd) > 1 else price
        chg = round((price - prev) / prev * 100, 2)

        return {
            "code": code, "name": name, "price": p, "market": "crypto",
            "price_usd": price_usd, "price_sar": price_sar,
            "stars": stars, "trend": trend,
            "above_daily": ad, "above_weekly": aw, "above_monthly": False,
            "score": score, "verdict": verdict, "bt": bt,
            "lb": round(p * 0.995, 4), "t1": t1, "t2": t2, "sl": sl,
            "rr": rr, "psl": psl, "ptp": ptp,
            "rsi": rsi_val, "adx": adx_val, "vr": 1,
            "liq": 0, "liq_unit": "م.$",
            "confidence": confidence, "chg": chg,
            "is_custom": False, "rank_score": 0,
        }
    except: return None

def run_scan(market, period="daily"):
    scan_state[market]["status"] = "scanning"
    scan_state[market]["progress"] = 0
    scan_state[market]["period"] = period
    custom = load_custom(); excl = custom.get("excluded", [])
    bdf = get_df(BENCHMARK.get(market, ""), "2y", "1d")

    if market == "tadawul":
        stocks = {**DEFAULT_TADAWUL, **custom.get("tadawul", {})}
    elif market == "us":
        stocks = {**DEFAULT_US, **custom.get("us", {})}
        # نمسح 80 سهم عشوائي كل مرة
        items = list(stocks.items())
        random.seed(int(time.time()) // 3600)
        random.shuffle(items)
        stocks = dict(items[:80])
    else:
        stocks = {**DEFAULT_CRYPTO, **custom.get("crypto", {})}

    stocks = {k: v for k, v in stocks.items() if k not in excl}
    total = len(stocks); scan_state[market]["total"] = total
    results = []; lock = threading.Lock(); done = [0]

    def scan_one(code, name):
        if market == "crypto":
            r = analyze_crypto(code, name, bdf, period)
        else:
            r = analyze(code, name, market, bdf, period)
        with lock:
            done[0] += 1
            scan_state[market]["progress"] = round(done[0] / total * 100)
            if r:
                r["is_custom"] = code in custom.get(market, {})
                results.append(r)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(scan_one, c, n) for c, n in stocks.items()]
        for f in as_completed(futs): pass

    # ترتيب وتصفية أفضل 5
    buy_results = [s for s in results if s["verdict"] == "BUY"]
    buy_results.sort(key=lambda x: (-x["score"], -x["confidence"], -x["rr"]))
    top5 = buy_results[:5]

    # لو ما في 5 BUY، نضيف WAIT
    if len(top5) < 5:
        wait_results = [s for s in results if s["verdict"] == "WAIT"]
        wait_results.sort(key=lambda x: (-x["score"], -x["confidence"]))
        top5 += wait_results[:5 - len(top5)]

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
    for i, s in enumerate(top5):
        s["rank_pos"] = i + 1
        s["medal"] = medals[i] if i < len(medals) else str(i + 1)

    scan_state[market]["data"] = top5
    scan_state[market]["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state[market]["status"] = "done"
    scan_state[market]["progress"] = 100
    scan_state[market]["total_scanned"] = len(results)


# ══ HTML ══
def build_card(s, idx, fav_codes=None):
    verdict = s["verdict"]
    chg = s.get("chg", 0)
    chg_cls = "up" if chg >= 0 else "down"
    chg_sign = "+" if chg >= 0 else ""
    stars = "★" * s["stars"] + "☆" * (3 - s["stars"])

    # رابط TradingView
    if s["market"] == "tadawul":
        tv_link = f'https://www.tradingview.com/chart/?symbol=TADAWUL%3A{s["code"]}'
    elif s["market"] == "us":
        tv_link = f'https://www.tradingview.com/chart/?symbol={s["code"]}'
    else:
        crypto_code = s["code"].replace("-USD", "USDT")
        tv_link = f'https://www.tradingview.com/chart/?symbol=BINANCE%3A{crypto_code}'

    # السعر
    if s["market"] == "us" and s.get("price_usd"):
        price_html = f'<span class="price-main">${s["price_usd"]}</span><span class="price-sub">{s["price_sar"]} ر.س</span>'
    elif s["market"] == "crypto" and s.get("price_usd"):
        price_usd = s["price_usd"]
        price_sar = s["price_sar"]
        if price_usd >= 1:
            price_html = f'<span class="price-main">${price_usd:,.2f}</span><span class="price-sub">{price_sar:,.2f} ر.س</span>'
        else:
            price_html = f'<span class="price-main">${price_usd}</span><span class="price-sub">{price_sar} ر.س</span>'
    else:
        price_html = f'<span class="price-main">{s["price"]} ر.س</span>'

    sfx = ".SR" if s["market"] == "tadawul" else ""

    # شرح الحكم
    if "مشروط" in s.get("bt", ""):
        badge_cls = "cond-badge"
        badge_txt = "شراء ⚠️"
        cond_reason = ""
        if s.get("rr", 0) < 1.3: cond_reason += "الربح أقل من المخاطرة بالمثالي · "
        if not s.get("above_monthly"): cond_reason += "الاتجاه الشهري ضعيف · "
        if s.get("score", 0) < 15: cond_reason += f'النقاط {s.get("score",0)}/20 · '
        cond_reason = cond_reason.rstrip(" · ")
        cond_html = f'<div class="cond-explain">⚠️ مشروط لأن: {cond_reason}</div>' if cond_reason else ""
    elif verdict == "BUY":
        badge_cls = "buy-badge"; badge_txt = "شراء"; cond_html = ""
    elif verdict == "WAIT":
        badge_cls = "wait-badge"; badge_txt = "انتظر"; cond_html = ""
    else:
        badge_cls = "avoid-badge"; badge_txt = "تجنب"; cond_html = ""

    conf = s.get("confidence", 0)
    conf_cls = "conf-high" if conf >= 70 else ("conf-mid" if conf >= 50 else "conf-low")
    trend_icon = {"استثمار": "📈", "سوينج": "↗️", "مضاربة": "⚡", "تجنب": "⬇️"}.get(s.get("trend", ""), "")

    # الوقت المتوقع
    atr_est = abs(s.get("t1", s["price"]) - s["price"])
    if atr_est > 0 and s["price"] > 0:
        days_est = max(1, round(atr_est / (s["price"] * 0.012)))
        if s.get("trend") == "استثمار": dur = f"{days_est*2}-{days_est*4} يوم"
        elif s.get("trend") == "سوينج": dur = f"{days_est}-{days_est*2} يوم"
        else: dur = f"{max(1,days_est-1)}-{days_est+2} يوم"
    else:
        dur = "—"

    # الأسعار الفعلية لـ TP وSL
    t1_price = s.get("t1", 0)
    t2_price = s.get("t2", 0)
    sl_price = s.get("sl", 0)
    ptp = s.get("ptp", 0)
    psl = s.get("psl", 0)

    # شرح الشارت المبسط
    chart_tips = []
    if s.get("above_daily") and s.get("above_weekly") and s.get("above_monthly"):
        chart_tips.append("✅ السعر فوق المتوسطات الثلاثة — الاتجاه صاعد في كل الإطارات")
    elif s.get("above_daily") and s.get("above_weekly"):
        chart_tips.append("✅ يومي وأسبوعي صاعدان — جيد لكن راقب الشهري")
    else:
        chart_tips.append("⚠️ الاتجاه غير مكتمل — كن حذراً")

    rsi = s.get("rsi", 50)
    if 40 <= rsi <= 60:
        chart_tips.append("✅ RSI في المنطقة المثالية — لا ذروة شراء ولا بيع")
    elif rsi > 70:
        chart_tips.append("⚠️ RSI مرتفع — السهم قد يكون مشبعاً بالشراء، انتظر تراجعاً")
    elif rsi < 40:
        chart_tips.append("⚠️ RSI منخفض — ضعف في الزخم")

    adx = s.get("adx", 0)
    if adx >= 25:
        chart_tips.append(f"✅ ADX {adx} — الاتجاه قوي وواضح")
    else:
        chart_tips.append(f"⚠️ ADX {adx} — السوق عرضي، الدخول محفوف بالمخاطر")

    if s.get("vr", 1) >= 1.5:
        chart_tips.append(f"✅ حجم تداول ×{s.get('vr',1)} — المشترين يدخلون بقوة")
    else:
        chart_tips.append("⚠️ حجم تداول عادي — لا يوجد اهتمام استثنائي")

    # جملة واحدة واضحة بدل قائمة تقنية
    score = s.get("score", 0)
    rsi = s.get("rsi", 50)
    adx = s.get("adx", 0)
    vr = s.get("vr", 1)
    ad = s.get("above_daily", False)
    aw = s.get("above_weekly", False)
    am = s.get("above_monthly", False)

    if score >= 15 and adx >= 25 and 40 <= rsi <= 65 and vr >= 1.3:
        main_msg = "🟢 الوضع ممتاز — السهم في اتجاه صاعد قوي وحجم الشراء يرتفع. الدخول مناسب الآن عند سعر الدخول المحدد."
    elif score >= 13 and ad and aw:
        if rsi > 65:
            main_msg = "🟡 الفرصة جيدة لكن السهم ارتفع مؤخراً — انتظر تراجعاً بسيطاً للسعر قريباً من سعر الدخول قبل تدخل."
        elif adx < 20:
            main_msg = "🟡 الإشارة جيدة لكن السوق عرضي حالياً — انتظر حركة واضحة وارتفاع في حجم التداول قبل الدخول."
        else:
            main_msg = "🟢 فرصة جيدة — الاتجاه اليومي والأسبوعي صاعدان. ادخل عند سعر الدخول المحدد."
    elif score >= 10:
        main_msg = "🟡 السهم يراقب فقط — لم تكتمل الشروط بعد. انتظر تحسن الإشارات قبل الدخول."
    else:
        main_msg = "🔴 ليس وقت الدخول — السهم في اتجاه هابط أو ضعيف. تجنب حالياً."

    # تحذير إضافي واضح
    warning = ""
    if rsi > 70:
        warning = "⚠️ تنبيه: السهم مشبع بالشراء (RSI {rsi}) — قد يتراجع قريباً. لو دخلت خفف الكمية."
    elif not am:
        warning = "⚠️ تنبيه: الاتجاه الشهري ليس صاعداً — الصفقة قصيرة المدى فقط."

    action_tip = f"👉 افتح الشارت: إذا السعر قريب من {s['lb']} والشموع خضراء = ادخل. إذا السعر فوقه بكثير = انتظر."

    tips_html = f'''<div class="tip-main">{main_msg}</div>
    {f'<div class="tip-warn">{warning}</div>' if warning else ""}
    <div class="tip-action">{action_tip}</div>
    <a href="{tv_link}" target="_blank" onclick="event.stopPropagation()" class="tv-btn">📊 افتح الشارت في TradingView</a>'''  

    # ══ حكم القرار ══
    if verdict == "BUY" and "مشروط" not in s.get("bt",""):
        if conf >= 75 and score >= 15 and adx >= 25:
            decision = "ENTER"
            dec_label = "ادخل الآن"
            dec_icon = "🟢"
            dec_color = "#16a34a"
            dec_bg = "#f0fdf4"
            dec_border = "#86efac"
            dec_reason = f"النقاط {score}/20 · ثقة {conf}% · اتجاه قوي"
        elif conf >= 60:
            decision = "ENTER"
            dec_label = "دخول جيد"
            dec_icon = "🟢"
            dec_color = "#16a34a"
            dec_bg = "#f0fdf4"
            dec_border = "#86efac"
            dec_reason = f"النقاط {score}/20 · ثقة {conf}%"
        else:
            decision = "WATCH"
            dec_label = "راقب وادخل"
            dec_icon = "🟡"
            dec_color = "#d97706"
            dec_bg = "#fffbeb"
            dec_border = "#fcd34d"
            dec_reason = f"النقاط {score}/20 · ثقة {conf}% · تحقق من الشارت أولاً"
    elif verdict == "BUY" and "مشروط" in s.get("bt",""):
        decision = "CAUTION"
        dec_label = "دخول بحذر"
        dec_icon = "🟡"
        dec_color = "#d97706"
        dec_bg = "#fffbeb"
        dec_border = "#fcd34d"
        dec_reason = f"فرصة لكن فيه تحفظ · راجع الشرح أدناه"
    elif verdict == "WAIT":
        decision = "WAIT"
        dec_label = "انتظر"
        dec_icon = "⏳"
        dec_color = "#6b7280"
        dec_bg = "#f9fafb"
        dec_border = "#d1d5db"
        dec_reason = f"النقاط {score}/20 · الشروط لم تكتمل بعد"
    else:
        decision = "AVOID"
        dec_label = "تجنب"
        dec_icon = "🔴"
        dec_color = "#dc2626"
        dec_bg = "#fef2f2"
        dec_border = "#fca5a5"
        dec_reason = f"النقاط {score}/20 · الاتجاه ضعيف أو هابط"

    return f"""
<div class="card card-{verdict.lower()}" onclick="toggleCard('{idx}')">
  <div class="decision-bar" style="background:{dec_bg};border-bottom:2px solid {dec_border};border-radius:14px 14px 0 0;padding:10px 16px;margin:-16px -16px 14px -16px;display:flex;align-items:center;justify-content:space-between;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:1.3rem;">{dec_icon}</span>
      <div>
        <div style="font-size:1rem;font-weight:800;color:{dec_color};">{dec_label}</div>
        <div style="font-size:0.68rem;color:#6b7280;margin-top:1px;">{dec_reason}</div>
      </div>
    </div>
    <div style="text-align:left;">
      <div style="font-size:1.2rem;font-weight:800;color:{dec_color};">{score}/20</div>
      <div style="font-size:0.62rem;color:#6b7280;">نقاط JRF</div>
    </div>
  </div>
  <div class="card-top">
    <div class="card-rank">{s.get("medal","")}</div>
    <div class="card-info">
      <div class="card-name">
        {s["name"]}
        <span class="card-code">{s["code"]}{sfx}</span>
        <a href="{tv_link}" target="_blank" onclick="event.stopPropagation()" class="tv-link" title="افتح في TradingView">📊</a>
      </div>
      <div class="card-meta">{trend_icon} {s.get("trend","")} · {stars} · ⏱ {dur}</div>
    </div>
    <div class="card-right">
      <div class="card-price">{price_html}</div>
      <div class="card-chg {chg_cls}">{chg_sign}{chg}%</div>
    </div>
  </div>
  <label class="fav-label" onclick="event.stopPropagation()">
    <input type="checkbox" class="fav-check" id="fav-{idx}"
      {'checked' if fav_codes and s['code'] in fav_codes else ''}
      onchange="toggleFav('{s['code']}','{s['market']}',this)">
    <span class="fav-txt">⭐ حفظ في المفضلة</span>
  </label>
  <div class="card-bar">
    <div class="card-bar-inner" style="width:{conf}%"></div>
  </div>
  <div class="card-summary">
    <span class="{badge_cls}">{badge_txt}</span>
    <span class="tag">🎯 TP1: {t1_price} <small>(+{ptp}%)</small></span>
    <span class="tag">🛡 SL: {sl_price} <small>(-{psl}%)</small></span>
    <span class="tag">R:R {s["rr"]}</span>
    <span class="{conf_cls}">ثقة {conf}%</span>
  </div>
  {cond_html}
  <div class="card-detail" id="det-{idx}" style="display:none">

    <div class="detail-grid">
      <div class="detail-box">
        <div class="detail-label">سعر الدخول</div>
        <div class="detail-val entry-val" onclick="copyVal(event,this)">{s["lb"]}</div>
      </div>
      <div class="detail-box">
        <div class="detail-label">هدف 1 💰</div>
        <div class="detail-val tp-val" onclick="copyVal(event,this)">{t1_price}<div class="sub-pct">+{ptp}%</div></div>
      </div>
      <div class="detail-box">
        <div class="detail-label">هدف 2 🚀</div>
        <div class="detail-val tp-val" onclick="copyVal(event,this)">{t2_price}<div class="sub-pct">+{round(ptp*2,1)}%</div></div>
      </div>
      <div class="detail-box">
        <div class="detail-label">وقف الخسارة 🛡</div>
        <div class="detail-val sl-val" onclick="copyVal(event,this)">{sl_price}<div class="sub-pct">-{psl}%</div></div>
      </div>
    </div>

    <div class="section-title">📊 إيش تشوف في الشارت قبل الدخول</div>
    <div class="chart-tips">{tips_html}
    </div>

    <div class="section-title">📈 خطة الخروج</div>
    <div class="exit-plan">
      <div class="exit-steps">
        <div class="exit-step"><span>عند الهدف 1 ({t1_price})</span><span class="exit-pct">بع 50%</span></div>
        <div class="exit-step"><span>عند الهدف 2 ({t2_price})</span><span class="exit-pct">بع 30%</span></div>
        <div class="exit-step"><span>الباقي</span><span class="exit-pct">اترك يجري 20%</span></div>
      </div>
    </div>

    <div class="indicators-row">
      <div class="ind-chip">RSI <strong>{s["rsi"]}</strong></div>
      <div class="ind-chip">ADX <strong>{s["adx"]}</strong></div>
      <div class="ind-chip">سيولة <strong>{s["liq"]} {s["liq_unit"]}</strong></div>
      <div class="ind-chip">حجم <strong>×{s["vr"]}</strong></div>
      <div class="ind-chip">وقت <strong>{dur}</strong></div>
    </div>

  </div>
</div>"""

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>جلال رادار</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Tajawal:wght@300;400;500;700;800&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #f0f4f8;
  --surface: #ffffff;
  --surface2: #f8fafc;
  --border: #e2e8f0;
  --border2: #cbd5e1;
  --text: #0f172a;
  --text2: #475569;
  --text3: #94a3b8;
  --accent: #2563eb;
  --accent2: #1d4ed8;
  --green: #16a34a;
  --green-bg: #f0fdf4;
  --green-border: #bbf7d0;
  --red: #dc2626;
  --red-bg: #fef2f2;
  --red-border: #fecaca;
  --yellow: #d97706;
  --yellow-bg: #fffbeb;
  --yellow-border: #fde68a;
  --gold: #f59e0b;
  --radius: 16px;
  --shadow: 0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.05);
  --shadow-lg: 0 4px 24px rgba(0,0,0,0.10);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Tajawal', sans-serif;
  min-height: 100vh;
  direction: rtl;
}
body::before {
  content: '';
  position: fixed;
  top: 0; left: 0; right: 0;
  height: 260px;
  background: linear-gradient(135deg, #1e40af 0%, #2563eb 50%, #0ea5e9 100%);
  z-index: 0;
}

.wrap { position: relative; z-index: 1; max-width: 780px; margin: 0 auto; padding: 0 16px 40px; }

/* ══ Header ══ */
.hdr {
  padding: 32px 0 24px;
  text-align: center;
  color: white;
}
.logo {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  margin-bottom: 4px;
}
.logo-en { font-size: 0.8rem; opacity: 0.7; letter-spacing: 2px; font-weight: 400; }
.mkt-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-top: 10px;
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 20px;
  padding: 4px 14px;
  font-size: 0.78rem;
  color: rgba(255,255,255,0.9);
  backdrop-filter: blur(10px);
}

/* ══ Tabs ══ */
.tabs-wrap {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 6px;
  display: flex;
  gap: 4px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
}
.tab-btn {
  flex: 1;
  padding: 10px;
  border: none;
  border-radius: 12px;
  background: transparent;
  color: var(--text2);
  font-family: 'Tajawal', sans-serif;
  font-size: 0.9rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}
.tab-btn.active {
  background: var(--accent);
  color: white;
  font-weight: 700;
  box-shadow: 0 2px 8px rgba(37,99,235,0.3);
}

/* ══ Period Tabs ══ */
.period-wrap {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}
.period-btn {
  flex: 1;
  padding: 8px;
  border: 1.5px solid var(--border);
  border-radius: 10px;
  background: var(--surface);
  color: var(--text2);
  font-family: 'Tajawal', sans-serif;
  font-size: 0.82rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.18s;
}
.period-btn.active {
  border-color: var(--accent);
  color: var(--accent);
  background: #eff6ff;
  font-weight: 700;
}

/* ══ Scan Button ══ */
.scan-wrap { text-align: center; margin-bottom: 20px; }
.scan-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--accent);
  color: white;
  border: none;
  border-radius: 50px;
  padding: 12px 32px;
  font-family: 'Tajawal', sans-serif;
  font-size: 1rem;
  font-weight: 700;
  cursor: pointer;
  transition: all 0.2s;
  box-shadow: 0 4px 16px rgba(37,99,235,0.35);
}
.scan-btn:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(37,99,235,0.45); }
.scan-btn:active { transform: translateY(0); }
.scan-ts { font-size: 0.72rem; color: var(--text3); margin-top: 8px; }

/* ══ Progress ══ */
.prog-wrap {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 16px;
  box-shadow: var(--shadow);
  display: none;
}
.prog-wrap.show { display: block; }
.prog-label { font-size: 0.82rem; color: var(--text2); margin-bottom: 8px; display: flex; justify-content: space-between; }
.prog-bar { height: 6px; background: var(--border); border-radius: 6px; overflow: hidden; }
.prog-fill { height: 100%; background: linear-gradient(90deg, var(--accent), #0ea5e9); border-radius: 6px; transition: width 0.5s; width: 0%; }

/* ══ Summary Bar ══ */
.summary-bar {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}
.sum-card {
  background: var(--surface);
  border-radius: 12px;
  padding: 12px;
  text-align: center;
  box-shadow: var(--shadow);
}
.sum-num { font-size: 1.6rem; font-weight: 800; line-height: 1; }
.sum-lbl { font-size: 0.68rem; color: var(--text3); margin-top: 3px; }
.sum-green .sum-num { color: var(--green); }
.sum-yellow .sum-num { color: var(--yellow); }
.sum-blue .sum-num { color: var(--accent); }
.sum-signal { font-size: 0.85rem !important; font-weight: 700 !important; }

/* ══ Card ══ */
.card {
  background: var(--surface);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 12px;
  box-shadow: var(--shadow);
  border: 1.5px solid var(--border);
  cursor: pointer;
  transition: all 0.2s;
}
.card:hover { box-shadow: var(--shadow-lg); transform: translateY(-1px); }
.card-buy { border-right: 4px solid var(--green); }
.card-wait { border-right: 4px solid var(--yellow); }
.card-avoid { border-right: 4px solid var(--red); }

.card-top { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.card-rank { font-size: 1.4rem; flex-shrink: 0; line-height: 1; margin-top: 2px; }
.card-info { flex: 1; min-width: 0; }
.card-name { font-size: 1rem; font-weight: 700; color: var(--text); }
.card-code { color: var(--text3); font-size: 0.72rem; font-weight: 400; margin-right: 6px; }
.card-meta { font-size: 0.76rem; color: var(--text2); margin-top: 2px; }
.card-right { text-align: left; flex-shrink: 0; }
.price-main { display: block; font-size: 1.05rem; font-weight: 800; color: var(--text); }
.price-sub { display: block; font-size: 0.68rem; color: var(--text3); margin-top: 1px; }
.card-chg { font-size: 0.75rem; font-weight: 600; margin-top: 3px; }
.up { color: var(--green); } .down { color: var(--red); }

/* ══ Confidence Bar ══ */
.card-bar { height: 3px; background: var(--border); border-radius: 3px; overflow: hidden; margin-bottom: 10px; }
.card-bar-inner { height: 100%; background: linear-gradient(90deg, var(--green), #22c55e); border-radius: 3px; }

/* ══ Summary Tags ══ */
.card-summary { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.buy-badge { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.cond-badge { background: #fff7ed; color: #c2410c; border: 1px solid #fed7aa; padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.wait-badge { background: var(--yellow-bg); color: var(--yellow); border: 1px solid var(--yellow-border); padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.avoid-badge { background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); padding: 2px 10px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; }
.tag { background: var(--surface2); border: 1px solid var(--border); color: var(--text2); padding: 2px 8px; border-radius: 8px; font-size: 0.68rem; }
.conf-high { color: var(--green); font-size: 0.72rem; font-weight: 700; margin-right: auto; }
.conf-mid { color: var(--yellow); font-size: 0.72rem; font-weight: 700; margin-right: auto; }
.conf-low { color: var(--text3); font-size: 0.72rem; margin-right: auto; }

/* ══ Detail ══ */
.card-detail { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.detail-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-bottom: 12px; }
.detail-box { background: var(--surface2); border-radius: 10px; padding: 10px; text-align: center; }
.detail-label { font-size: 0.6rem; color: var(--text3); margin-bottom: 4px; }
.detail-val { font-size: 0.85rem; font-weight: 700; cursor: pointer; border-radius: 6px; padding: 2px 4px; transition: background 0.15s; }
.detail-val:hover { background: var(--border); }
.entry-val { color: var(--text); } .tp-val { color: var(--green); } .sl-val { color: var(--red); }
.small-pct { font-size: 0.6rem; font-weight: 400; }

.indicators-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px; }
.ind-chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 8px; padding: 4px 10px; font-size: 0.7rem; color: var(--text2); }
.ind-chip strong { color: var(--text); }

.ma-row { display: flex; gap: 6px; margin-bottom: 12px; }
.ma-tag { padding: 3px 10px; border-radius: 8px; font-size: 0.68rem; font-weight: 600; }
.ma-ok { background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }
.ma-no { background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }

.exit-plan { background: #f8fafc; border: 1px solid var(--border); border-radius: 10px; padding: 10px 14px; }
.exit-title { font-size: 0.72rem; font-weight: 700; color: var(--text2); margin-bottom: 8px; }
.exit-steps { display: flex; flex-direction: column; gap: 5px; }
.exit-step { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--text2); }
.exit-pct { font-weight: 700; color: var(--accent); }

/* ══ Empty State ══ */
.empty { text-align: center; padding: 60px 20px; }
.empty-icon { font-size: 3rem; margin-bottom: 14px; }
.empty-txt { color: var(--text2); font-size: 0.9rem; }

/* ══ Toast ══ */
.toast { position: fixed; bottom: 24px; left: 50%; transform: translateX(-50%); background: var(--text); color: white; padding: 8px 18px; border-radius: 20px; font-size: 0.78rem; opacity: 0; transition: opacity 0.25s; z-index: 200; pointer-events: none; white-space: nowrap; }
.toast.show { opacity: 1; }

/* ══ Loading ══ */
.lo { position: fixed; inset: 0; background: rgba(240,244,248,0.92); display: none; flex-direction: column; align-items: center; justify-content: center; z-index: 100; backdrop-filter: blur(4px); }
.lo.show { display: flex; }
.lo-spin { width: 48px; height: 48px; border: 3px solid var(--border); border-top-color: var(--accent); border-radius: 50%; animation: spin 0.8s linear infinite; margin-bottom: 16px; }
@keyframes spin { to { transform: rotate(360deg); } }
.lo-txt { color: var(--text); font-size: 0.95rem; font-weight: 600; }
.lo-sub { color: var(--text2); font-size: 0.75rem; margin-top: 6px; }
.lo-bar { width: 200px; height: 4px; background: var(--border); border-radius: 4px; margin-top: 14px; overflow: hidden; }
.lo-fill { height: 100%; background: var(--accent); border-radius: 4px; transition: width 0.5s; }

/* ══ Footer ══ */
.footer { text-align: center; font-size: 0.68rem; color: var(--text3); padding: 20px 0; }

@media(max-width:600px) {
  .detail-grid { grid-template-columns: repeat(2,1fr); }
  .summary-bar { grid-template-columns: repeat(3,1fr); }
  .logo { font-size: 1.8rem; }
}
</style>
</head>
<body>
<div class="lo" id="lo">
  <div class="lo-spin"></div>
  <div class="lo-txt">🔍 جاري المسح...</div>
  <div class="lo-sub" id="lo-sub">يتحقق من البيانات</div>
  <div class="lo-bar"><div class="lo-fill" id="lo-fill" style="width:0%"></div></div>
</div>
<div class="toast" id="toast"></div>

<div class="wrap">
  <div class="hdr">
    <div class="logo">⚡ جلال رادار</div>
    <div class="logo-en">JALAL RADAR v5 · HALAL EDITION</div>
    <div class="mkt-badge" id="sess-bar">__SESSION__</div>
  </div>

  <div class="tabs-wrap">
    <button class="tab-btn active" id="tb-tadawul" onclick="switchTab('tadawul')">🇸🇦 تاسي</button>
    <button class="tab-btn" id="tb-us" onclick="switchTab('us')">🇺🇸 أمريكي</button>
    <button class="tab-btn" id="tb-crypto" onclick="switchTab('crypto')">💰 عملات</button>
  </div>

  __ADD_PANEL__
  __TAB_CONTENTS__
</div>

<div class="footer">جلال رادار v5 · للأغراض التعليمية فقط · ليس توصية استثمارية</div>

<script>
var aTab = 'tadawul', aPeriod = {tadawul:'daily', us:'daily', crypto:'daily'};
var scanPoll = null;

function switchTab(t) {
  aTab = t;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-c').forEach(c => c.style.display = 'none');
  document.getElementById('tb-' + t).classList.add('active');
  document.getElementById('tc-' + t).style.display = 'block';
}

function setPeriod(p) {
  aPeriod[aTab] = p;
  document.querySelectorAll('#period-' + aTab + ' .period-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.p === p);
  });
}

function startScan(mkt) {
  if (mkt) aTab = mkt;
  document.getElementById('lo').classList.add('show');
  document.getElementById('lo-sub').textContent = 'يتحقق من البيانات...';
  document.getElementById('lo-fill').style.width = '0%';
  fetch('/scan?market=' + aTab + '&period=' + aPeriod[aTab])
    .then(r => r.json())
    .then(() => { scanPoll = setInterval(pollProgress, 1500); });
}

function pollProgress() {
  fetch('/status?market=' + aTab).then(r => r.json()).then(d => {
    var pct = d.progress || 0;
    document.getElementById('lo-fill').style.width = pct + '%';
    document.getElementById('lo-sub').textContent = 'تم مسح ' + pct + '% — ' + d.total + ' سهم';
    if (d.status === 'done') {
      clearInterval(scanPoll);
      sessionStorage.setItem('activeTab', aTab);
      window.location.reload();
    }
  });
}

(function() {
  var saved = sessionStorage.getItem('activeTab');
  if (saved) { sessionStorage.removeItem('activeTab'); switchTab(saved); }
})();

function toggleCard(i) {
  var det = document.getElementById('det-' + i);
  if (!det) return;
  det.style.display = det.style.display === 'none' ? 'block' : 'none';
}

function copyVal(e, el) {
  e.stopPropagation();
  var t = el.textContent.trim().split('\n')[0].trim();
  navigator.clipboard.writeText(t).then(() => {
    el.style.background = '#dcfce7';
    showToast('✓ تم النسخ: ' + t);
    setTimeout(() => el.style.background = '', 1200);
  });
}

function showToast(m) {
  var t = document.getElementById('toast');
  t.textContent = m; t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 2000);
}

function addStock() {
  var code = document.getElementById('nCode').value.trim().toUpperCase();
  var name = document.getElementById('nName').value.trim();
  var mkt = document.getElementById('nMkt').value;
  if (!code || !name) { showToast('أدخل الرمز والاسم'); return; }
  fetch('/add_stock', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({code, name, market: mkt})})
  .then(r => r.json()).then(d => {
    if (d.ok) { showToast('تمت الإضافة ✓'); setTimeout(() => location.reload(), 800); }
  });
}
// ══ المفضلة ══
function toggleFav(code, market, cb) {
  fetch('/favorites/toggle', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({code, market})
  }).then(r => r.json()).then(d => {
    if (d.ok) {
      showToast(d.added ? '⭐ تمت الإضافة للمفضلة' : '✕ تمت الإزالة من المفضلة');
      // نحدث قسم المفضلة بدون reload
      setTimeout(() => location.reload(), 600);
    }
  });
}

// ══ إضافة سهم — الاسم اختياري ══
function addStock() {
  var code = document.getElementById('nCode').value.trim().toUpperCase();
  var name = document.getElementById('nName').value.trim();
  var mkt = document.getElementById('nMkt').value;
  // يقبل رمز أو اسم — واحد يكفي
  if (!code && !name) { showToast('أدخل الرمز أو الاسم'); return; }
  // لو ما في رمز، نستخدم الاسم كرمز مؤقت
  if (!code) code = name.replace(/\s+/g, '_').toUpperCase();
  showToast('جاري إضافة ' + code + '...');
  fetch('/add_stock', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({code, name, market: mkt})})
  .then(r => r.json()).then(d => {
    if (d.ok) {
      showToast('تمت إضافة ' + (d.name || code) + ' ✓');
      setTimeout(() => location.reload(), 900);
    } else {
      showToast('خطأ في الإضافة');
    }
  });
}

function exclStock(code) {
  fetch('/exclude', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({code})})
  .then(r => r.json()).then(d => { if (d.ok) location.reload(); });
}
</script>
</body>
</html>"""


@app.route("/")
def index():
    from datetime import datetime
    now = datetime.utcnow()
    h = now.hour; wd = now.weekday()
    if wd >= 5: sess = "🔴 السوق مغلق — عطلة"
    elif 13 <= h < 20: sess = "🟢 London + NY مفتوح"
    elif 8 <= h < 13: sess = "🟡 London مفتوح"
    else: sess = "🔴 السوق مغلق"

    # ══ لوحة إضافة أسهم ══
    custom = load_custom()
    excl = custom.get("excluded", [])
    excl_html = "".join(f'<span class="excl-tag" onclick="exclStock(\'{c}\')" title="اضغط لإزالة الاستبعاد">{c} ✕</span>' for c in excl)
    add_panel = f'''<div class="add-panel">
      <div class="add-title">➕ إضافة سهم للتحليل</div>
      <div class="add-row">
        <input class="add-inp" id="nCode" placeholder="الرمز أو الاسم (مثال: AAPL أو 2222)">
        <input class="add-inp" id="nName" placeholder="الاسم (اختياري)">
        <select class="add-inp" id="nMkt" style="max-width:120px;">
          <option value="tadawul">🇸🇦 تاسي</option>
          <option value="us">🇺🇸 أمريكي</option>
          <option value="crypto">💰 عملات</option>
        </select>
        <button class="add-btn-green" onclick="addStock()">➕ أضف</button>
      </div>
      {('<div class="excl-list">مستبعدون: ' + excl_html + '</div>') if excl else ''}
    </div>'''

    tab_contents = ""
    for mkt, label, flag in [("tadawul","تاسي","🇸🇦"), ("us","الأمريكي","🇺🇸"), ("crypto","العملات","💰")]:
        data = scan_state[mkt]["data"] or []
        last = scan_state[mkt]["last_scan"] or ""
        total_scanned = scan_state[mkt].get("total_scanned", 0)
        period = scan_state[mkt].get("period", "daily")
        style = 'display:block' if mkt == "tadawul" else 'display:none'

        buy_n = sum(1 for s in data if s["verdict"] == "BUY")
        wait_n = sum(1 for s in data if s["verdict"] == "WAIT")

        period_btns = ""
        for pid, plabel in [("daily","يومي"), ("weekly","أسبوعي"), ("monthly","شهري")]:
            active = "active" if period == pid else ""
            period_btns += f'<button class="period-btn {active}" data-p="{pid}" onclick="setPeriod(\'{pid}\')">{plabel}</button>'

        # تحميل المفضلة لهذا السوق
        favs = load_favorites()
        fav_codes = favs.get(mkt, [])

        # ══ قسم المفضلة ══
        fav_section = ""
        if fav_codes:
            fav_section = f'''<div class="fav-section">
              <div class="fav-title">⭐ مفضلتك ({len(fav_codes)} سهم) — يتم تحليلها في كل مسح</div>
              <div class="fav-chips">'''
            for fc in fav_codes:
                fav_section += f'<span class="fav-chip">✅ {fc} <span class="fav-rm" onclick="toggleFav(\'{fc}\',\'{mkt}\')" title="إزالة">✕</span></span>'
            fav_section += '</div></div>'

        cards_html = ""
        if data:
            for i, s in enumerate(data):
                cards_html += build_card(s, f"{mkt}-{i}", fav_codes)
        else:
            cards_html = f'''<div class="empty">
              <div class="empty-icon">📡</div>
              <div class="empty-txt">اضغط مسح لتحليل {label}</div>
            </div>'''

        scan_info = f'<div class="scan-ts">آخر مسح: {last} · رُشّح من {total_scanned} سهم</div>' if last else ""

        summary = ""
        if data:
            summary = f'''<div class="summary-bar">
              <div class="sum-card sum-green"><div class="sum-num">{buy_n}</div><div class="sum-lbl">شراء</div></div>
              <div class="sum-card sum-yellow"><div class="sum-num">{wait_n}</div><div class="sum-lbl">انتظر</div></div>
              <div class="sum-card sum-blue"><div class="sum-num">{len(data)}</div><div class="sum-lbl">مرشّح</div></div>
            </div>'''

        tab_contents += f'''<div class="tab-c" id="tc-{mkt}" style="{style}">
          <div class="period-wrap" id="period-{mkt}">{period_btns}</div>
          <div class="scan-wrap">
            <button class="scan-btn" onclick="startScan('{mkt}')">🔍 مسح {flag} {label}</button>
            {scan_info}
          </div>
          {summary}
          {fav_section}
          {cards_html}
        </div>'''

    html = HTML_TEMPLATE.replace("__SESSION__", sess).replace("__TAB_CONTENTS__", tab_contents).replace("__ADD_PANEL__", add_panel)
    return html


@app.route("/scan")
def scan():
    m = request.args.get("market", "tadawul")
    period = request.args.get("period", "daily")
    if scan_state[m]["status"] == "scanning":
        return jsonify({"status": "already_running"})
    t = threading.Thread(target=run_scan, args=(m, period))
    t.daemon = True; t.start()
    return jsonify({"status": "started"})

@app.route("/status")
def status():
    m = request.args.get("market", "tadawul")
    return jsonify({
        "status": scan_state[m]["status"],
        "progress": scan_state[m].get("progress", 0),
        "total": scan_state[m].get("total", 0)
    })

@app.route("/add_stock", methods=["POST"])
def add_stock():
    d = request.get_json()
    code = d.get("code","").strip().upper()
    name = d.get("name","").strip()
    mkt = d.get("market","tadawul")
    if not code: return jsonify({"ok": False, "msg": "أدخل الرمز"})
    # لو ما في اسم، نجيبه تلقائياً من yfinance
    if not name:
        try:
            ticker_str = code + ".SR" if mkt == "tadawul" else code
            tk = yf.Ticker(ticker_str)
            info = tk.fast_info
            name = getattr(info, 'company_name', None) or code
            if not name or name == code:
                # جرب longName
                full_info = tk.info
                name = full_info.get("longName") or full_info.get("shortName") or code
        except:
            name = code
    c = load_custom(); c.setdefault(mkt, {})[code] = name
    if code in c.get("excluded", []): c["excluded"].remove(code)
    save_custom(c); return jsonify({"ok": True, "name": name})

@app.route("/exclude", methods=["POST"])
def exclude():
    code = request.get_json().get("code", "")
    c = load_custom()
    if code not in c.get("excluded", []): c.setdefault("excluded", []).append(code)
    save_custom(c); return jsonify({"ok": True})

@app.route("/favorites", methods=["GET"])
def favorites_get():
    return jsonify(load_favorites())

@app.route("/favorites/toggle", methods=["POST"])
def favorites_toggle():
    d = request.get_json()
    code = d.get("code", "").strip().upper()
    market = d.get("market", "tadawul")
    favs = load_favorites()
    mkt_list = favs.get(market, [])
    if code in mkt_list:
        mkt_list.remove(code)
        added = False
    else:
        mkt_list.append(code)
        added = True
    favs[market] = mkt_list
    save_favorites(favs)
    return jsonify({"ok": True, "added": added})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  ⚡ جلال رادار v5.0 — Clean & Smart")
    print(f"  تاسي: {len(DEFAULT_TADAWUL)} سهم")
    print(f"  أمريكي: {len(DEFAULT_US)} سهم")
    print(f"  عملات: {len(DEFAULT_CRYPTO)} عملة")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
