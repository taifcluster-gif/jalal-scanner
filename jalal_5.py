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

# ══ إعدادات Alpaca ══
ALPACA_CONFIG_FILE = os.path.join(os.getcwd(), "alpaca_config.json")

def load_alpaca():
    # نقرأ من Environment Variables أولاً
    env_key = os.environ.get("ALPACA_KEY", "")
    env_secret = os.environ.get("ALPACA_SECRET", "")
    defaults = {
        "key": env_key, "secret": env_secret,
        "endpoint": "https://paper-api.alpaca.markets/v2",
        "enabled": bool(env_key), "max_position_usd": 500,
        "max_daily_trades": 5, "daily_loss_limit": 3.0,
        "auto_buy": False, "auto_sell": True
    }
    if os.path.exists(ALPACA_CONFIG_FILE):
        with open(ALPACA_CONFIG_FILE, "r") as f:
            saved = json.load(f)
            # نستخدم الـ env vars لو الملف ما عنده keys
            if not saved.get("key") and env_key:
                saved["key"] = env_key
            if not saved.get("secret") and env_secret:
                saved["secret"] = env_secret
            if not saved.get("enabled") and env_key:
                saved["enabled"] = True
            return {**defaults, **saved}
    return defaults

def save_alpaca(cfg):
    # نحفظ الإعدادات مع الـ keys
    with open(ALPACA_CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)
    # لو في env vars نحدثها
    if cfg.get("key"):
        os.environ["ALPACA_KEY"] = cfg["key"]
    if cfg.get("secret"):
        os.environ["ALPACA_SECRET"] = cfg["secret"]

def alpaca_request(method, path, data=None):
    cfg = load_alpaca()
    if not cfg.get("key"): return None
    import urllib.request, urllib.parse
    url = cfg["endpoint"].rstrip("/") + path
    headers = {
        "APCA-API-KEY-ID": cfg["key"],
        "APCA-API-SECRET-KEY": cfg["secret"],
        "Content-Type": "application/json"
    }
    try:
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}

def get_alpaca_account():
    return alpaca_request("GET", "/account")

def place_order(symbol, qty, side, order_type="market", limit_price=None, stop_price=None):
    data = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": order_type,
        "time_in_force": "gtc"
    }
    if limit_price: data["limit_price"] = str(limit_price)
    if stop_price: data["stop_price"] = str(stop_price)
    return alpaca_request("POST", "/orders", data)

def get_positions():
    return alpaca_request("GET", "/positions") or []

def get_orders(status="open"):
    return alpaca_request("GET", f"/orders?status={status}&limit=50") or []

def close_position(symbol):
    return alpaca_request("DELETE", f"/positions/{symbol}")

# ══ تتبع الصفقات ══
TRADES_FILE = os.path.join(os.getcwd(), "auto_trades.json")

def auto_trade_on_scan(signals):
    """تنفيذ صفقات تلقائية بناءً على إشارات الرادار"""
    cfg = load_alpaca()
    if not cfg.get("enabled") or not cfg.get("auto_buy"): return
    if not cfg.get("key"): return

    # نتحقق من عدد الصفقات اليومية
    today = datetime.now().strftime("%Y-%m-%d")
    trades = load_trades()
    today_trades = [t for t in trades if t.get("time","").startswith(today) and t.get("source") == "auto_radar"]
    max_daily = int(cfg.get("max_daily_trades", 5))
    if len(today_trades) >= max_daily: return

    # نتحقق من الخسارة اليومية
    acc = get_alpaca_account()
    if not acc or "error" in acc: return

    for signal in signals:
        if signal.get("market") != "us": continue
        if signal.get("verdict") != "BUY": continue
        if signal.get("confidence", 0) < 70: continue
        if "مشروط" in signal.get("bt", ""): continue

        # نتحقق ما اشترينا نفس السهم اليوم
        already = [t for t in today_trades if t.get("symbol") == signal.get("code")]
        if already: continue

        # نحسب الكمية
        max_usd = float(cfg.get("max_position_usd", 500))
        price = float(signal.get("price", 0))
        if price <= 0: continue
        qty = max(1, int(max_usd / price))

        # نرسل أمر شراء
        limit_price = signal.get("lb")
        result = place_order(signal["code"], qty, "buy", "limit", limit_price)
        if not result or "error" in result: continue

        save_trade({
            "symbol": signal["code"], "name": signal.get("name",""),
            "side": "buy", "qty": qty, "order_type": "limit",
            "limit_price": limit_price, "tp1": signal.get("t1"),
            "sl": signal.get("sl"), "rr": signal.get("rr"),
            "order_id": result.get("id",""),
            "status": result.get("status",""),
            "score": signal.get("score",0),
            "confidence": signal.get("confidence",0),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "source": "auto_radar"
        })

        if len(today_trades) + 1 >= max_daily: break

def check_daily_loss_limit():
    """تحقق من حد الخسارة اليومي وأوقف التداول لو تجاوزناه"""
    cfg = load_alpaca()
    if not cfg.get("enabled"): return False
    acc = get_alpaca_account()
    if not acc or "error" in acc: return False
    daily_pnl = float(acc.get("equity", 100000)) - 100000  # مقارنة بالبداية
    equity = float(acc.get("equity", 100000))
    loss_limit = float(cfg.get("daily_loss_limit", 3.0)) / 100
    if equity < 100000 * (1 - loss_limit):
        return True  # تجاوزنا الحد
    return False

def monitor_positions():
    """مراقبة المراكز المفتوحة وبيع عند TP أو SL"""
    cfg = load_alpaca()
    if not cfg.get("enabled") or not cfg.get("auto_sell"): return
    if not cfg.get("key"): return

    # تحقق من حد الخسارة
    if check_daily_loss_limit():
        # أغلق كل المراكز
        positions = get_positions()
        if isinstance(positions, list):
            for pos in positions:
                close_position(pos.get("symbol",""))
        return

    positions = get_positions()
    if not isinstance(positions, list): return
    trades = load_trades()

    for pos in positions:
        symbol = pos.get("symbol","")
        current = float(pos.get("current_price", 0))
        if current <= 0: continue

        # نجد الصفقة المقابلة
        matching = [t for t in trades if t.get("symbol") == symbol and t.get("side") == "buy" and t.get("source") == "auto_radar"]
        if not matching: continue

        trade = matching[-1]
        tp1 = float(trade.get("tp1") or 0)
        sl = float(trade.get("sl") or 0)

        should_sell = False
        reason = ""
        if tp1 > 0 and current >= tp1:
            should_sell = True; reason = "TP1 ✅"
        elif sl > 0 and current <= sl:
            should_sell = True; reason = "SL 🛡"

        if should_sell:
            result = close_position(symbol)
            if result and "error" not in result:
                save_trade({
                    "symbol": symbol, "side": "sell",
                    "qty": pos.get("qty","0"), "order_type": "market",
                    "reason": reason, "exit_price": current,
                    "pnl": pos.get("unrealized_pl","0"),
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "source": "auto_radar"
                })

# مراقب دوري كل دقيقة
def auto_scan_and_trade():
    """مسح تلقائي عند افتتاح السوق"""
    cfg = load_alpaca()
    if not cfg.get("enabled"): return
    bdf = get_df(BENCHMARK.get("us",""), "2y", "1d")
    stocks = {**DEFAULT_US}
    import random, time as _time
    items = list(stocks.items())
    random.seed(int(_time.time()) // 3600)
    random.shuffle(items)
    stocks = dict(items[:80])
    results = []
    lock = threading.Lock()
    done = [0]
    total = len(stocks)

    def scan_one(code, name):
        r = analyze(code, name, "us", bdf, "daily")
        with lock:
            done[0] += 1
            if r: results.append(r)

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = [ex.submit(scan_one, c, n) for c, n in stocks.items()]
        for f in as_completed(futs): pass

    buy_results = [s for s in results if s["verdict"] == "BUY"]
    buy_results.sort(key=lambda x: (-x["score"], -x["confidence"]))
    top5 = buy_results[:5]
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
    for i, s in enumerate(top5):
        s["rank_pos"] = i+1
        s["medal"] = medals[i] if i < len(medals) else str(i+1)

    scan_state["us"]["data"] = top5
    scan_state["us"]["last_scan"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state["us"]["total_scanned"] = len(results)

    # تداول تلقائي
    try: auto_trade_on_scan(top5)
    except: pass

def start_monitor():
    def loop():
        import time as _time
        last_scan_day = None
        while True:
            try:
                # مراقبة المراكز كل دقيقة
                monitor_positions()

                # مسح تلقائي عند افتتاح السوق (9:30 AM ET = 16:30 KSA = 13:30 UTC)
                now = datetime.utcnow()
                weekday = now.weekday()  # 0=Monday, 4=Friday
                hour = now.hour
                minute = now.minute
                today = now.strftime("%Y-%m-%d")

                # افتتاح السوق: الاثنين-الجمعة 13:30-13:35 UTC
                if (weekday < 5 and hour == 13 and 30 <= minute <= 35
                        and last_scan_day != today):
                    last_scan_day = today
                    try:
                        auto_scan_and_trade()
                        print(f"✅ مسح تلقائي {today}")
                    except Exception as e:
                        print(f"❌ خطأ في المسح: {e}")

            except: pass
            _time.sleep(60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()

def load_trades():
    if os.path.exists(TRADES_FILE):
        with open(TRADES_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_trade(trade):
    trades = load_trades()
    trades.append(trade)
    with open(TRADES_FILE, "w", encoding="utf-8") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2)

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

def analyze(code, name, market, bdf=None, period="daily", no_filter=False):
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

        # ══ فلترات الجودة (تُتجاوز لو no_filter=True) ══
        if not no_filter:
            # 1. فلتر السعر
            if market == "us" and price < 5.0: return None
            if market == "tadawul" and price < 1.0: return None

            # 2. فلتر السيولة اليومية
            try:
                avg_vol = float(sma(dfd["Volume"], 20).iloc[-1])
                daily_liquidity = price * avg_vol
                if market == "us" and daily_liquidity < 5_000_000: return None
                if market == "tadawul" and daily_liquidity < 10_000_000: return None
            except: pass

            # 3. فلتر Market Cap
            try:
                tk = yf.Ticker(ticker)
                info = tk.fast_info
                mkt_cap = getattr(info, 'market_cap', None)
                if mkt_cap:
                    if market == "us" and mkt_cap < 500_000_000: return None
                    if market == "tadawul" and mkt_cap < 500_000_000: return None
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

def analyze_crypto(code, name, bdf=None, period="daily", no_filter=False):
    try:
        if period == "weekly":
            dfd = get_df(code, "5y", "1wk")
        elif period == "monthly":
            dfd = get_df(code, "10y", "1mo")
        else:
            dfd = get_df(code, "2y", "1d")

        if dfd.empty or len(dfd) < 30: return None

        price = float(dfd["Close"].iloc[-1])

        # ══ فلترات جودة الكريبتو (تُتجاوز لو no_filter=True) ══
        if price < 0.000001: return None

        if not no_filter:
            try:
                avg_vol = float(sma(dfd["Volume"], 20).iloc[-1])
                daily_liq_usd = price * avg_vol
                if daily_liq_usd < 1_000_000: return None
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

    # تداول تلقائي لو مفعّل
    if market == "us":
        try: auto_trade_on_scan(top5)
        except: pass


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
    {f'<button class="btn-quick-trade" onclick="quickTrade(event,\'{s["code"]}\',\'{s["name"]}\',{s["price"]},{t1_price},{sl_price})">⚡ تداول</button>' if s["market"]=="us" and verdict=="BUY" else ""}
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
.btn-quick-trade { background: linear-gradient(135deg, #16a34a, #15803d); color: white; border: none; padding: 3px 10px; border-radius: 8px; font-size: 0.7rem; font-weight: 700; cursor: pointer; font-family: 'Tajawal', sans-serif; margin-right: auto; box-shadow: 0 2px 6px rgba(22,163,74,0.3); }
.btn-quick-trade:hover { transform: translateY(-1px); box-shadow: 0 4px 10px rgba(22,163,74,0.4); }

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

/* ══ قسم التداول ══ */
.trading-wrap { background: var(--surface); border-radius: var(--radius); padding: 18px; margin-bottom: 16px; box-shadow: var(--shadow); border: 1.5px solid var(--border); }
.trading-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
.trading-title { font-size: 0.95rem; font-weight: 800; color: var(--text); }
.trading-status { display: flex; align-items: center; gap: 6px; font-size: 0.72rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; }
.status-live { background: var(--green); box-shadow: 0 0 6px var(--green); }
.status-off { background: var(--text3); }
.account-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-bottom: 14px; }
.acc-card { background: var(--surface2); border-radius: 10px; padding: 10px; text-align: center; }
.acc-val { font-size: 1rem; font-weight: 800; color: var(--text); }
.acc-lbl { font-size: 0.62rem; color: var(--text3); margin-top: 3px; }
.trade-form { background: var(--surface2); border-radius: 12px; padding: 14px; margin-bottom: 14px; }
.trade-form-title { font-size: 0.78rem; font-weight: 700; color: var(--text2); margin-bottom: 10px; }
.trade-row { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
.trade-inp { flex: 1; min-width: 80px; background: white; border: 1.5px solid var(--border); color: var(--text); padding: 8px 10px; border-radius: 8px; font-family: 'Tajawal', sans-serif; font-size: 0.82rem; }
.trade-inp:focus { outline: none; border-color: var(--accent); }
.btn-buy { background: var(--green); color: white; border: none; padding: 8px 18px; border-radius: 8px; font-family: 'Tajawal', sans-serif; font-size: 0.82rem; font-weight: 700; cursor: pointer; }
.btn-sell { background: var(--red); color: white; border: none; padding: 8px 18px; border-radius: 8px; font-family: 'Tajawal', sans-serif; font-size: 0.82rem; font-weight: 700; cursor: pointer; }
.positions-list { margin-top: 12px; }
.pos-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; background: var(--surface2); border-radius: 10px; margin-bottom: 6px; border: 1px solid var(--border); }
.pos-sym { font-weight: 700; font-size: 0.88rem; }
.pos-info { font-size: 0.68rem; color: var(--text2); margin-top: 2px; }
.btn-close-pos { background: none; border: 1px solid var(--red-border); color: var(--red); padding: 4px 10px; border-radius: 6px; font-size: 0.7rem; cursor: pointer; font-family: 'Tajawal', sans-serif; }
.settings-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 10px; }
.setting-row { display: flex; flex-direction: column; gap: 4px; }
.setting-lbl { font-size: 0.68rem; color: var(--text3); }
.toggle-wrap { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.toggle-lbl { font-size: 0.75rem; color: var(--text2); cursor: pointer; }
.save-cfg { background: var(--accent); color: white; border: none; padding: 8px 20px; border-radius: 8px; font-family: 'Tajawal', sans-serif; font-size: 0.82rem; font-weight: 700; cursor: pointer; width: 100%; margin-top: 6px; }
.trade-log { max-height: 200px; overflow-y: auto; }
.log-item { display: flex; justify-content: space-between; padding: 6px 8px; border-radius: 6px; margin-bottom: 4px; font-size: 0.7rem; }
.log-buy { background: var(--green-bg); color: var(--green); }
.log-sell { background: var(--red-bg); color: var(--red); }
.tabs-trading { display: flex; gap: 4px; margin-bottom: 12px; }
.tab-trade-btn { flex: 1; padding: 7px; border: 1.5px solid var(--border); border-radius: 8px; background: var(--surface); color: var(--text2); font-family: 'Tajawal', sans-serif; font-size: 0.78rem; cursor: pointer; }
.tab-trade-btn.active { background: var(--accent); color: white; border-color: var(--accent); font-weight: 700; }

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

  __TRADING__
  __ADD_PANEL__
  <div id="instant-result" style="display:none;max-width:780px;margin:0 auto 16px;padding:0 16px;"></div>
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

function analyzeNow() {
  var code = document.getElementById('nCode').value.trim().toUpperCase();
  var name = document.getElementById('nName').value.trim();
  var mkt = document.getElementById('nMkt').value;
  if (!code && !name) { showToast('أدخل الرمز أولاً'); return; }
  if (!code) code = name.replace(/ +/g, '_').toUpperCase();
  var res = document.getElementById('instant-result');
  res.style.display = 'block';
  res.innerHTML = '<div class=\"instant-wrap\"><div class=\"instant-title\">جاري تحليل ' + code + '...</div></div>';
  var period = 'daily';
  var pb = document.querySelector('.period-btn.active');
  if (pb) period = pb.dataset.p;
  var noFilter = document.getElementById('noFilter') ? document.getElementById('noFilter').checked : false;
  fetch('/analyze_one', {method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({code:code,name:name,market:mkt,period:period,no_filter:noFilter})
  }).then(function(r){return r.json();}).then(function(d){
    if (!d.ok) {
      res.innerHTML = '<div class=\"instant-wrap\"><div style=\"color:red;padding:10px;\">❌ ' + (d.msg||'تعذر التحليل') + '</div></div>';
      return;
    }
    var s=d.result, h='';
    var sfx=s.market==='tadawul'?'.SR':'';
    var tvLink=s.market==='tadawul'?'https://www.tradingview.com/chart/?symbol=TADAWUL%3A'+s.code:'https://www.tradingview.com/chart/?symbol='+s.code;
    var priceHtml=s.market==='tadawul'?(s.price+' ر.س'):('$'+s.price_usd+' · '+s.price_sar+' ر.س');
    var dc=s.dec_color||'#6b7280', db=s.dec_bg||'#f9fafb', dbd=s.dec_border||'#d1d5db';
    h='<div class=\"instant-wrap\">';
    h+='<div class=\"instant-title\">✅ '+s.name+' ('+s.code+sfx+')</div>';
    h+='<div style=\"background:'+db+';border:1.5px solid '+dbd+';border-radius:12px;padding:12px;margin-bottom:12px;display:flex;justify-content:space-between;\">';
    h+='<div style=\"display:flex;align-items:center;gap:8px;\"><span style=\"font-size:1.3rem;\">'+(s.dec_icon||'⏳')+'</span>';
    h+='<div><div style=\"font-size:1rem;font-weight:800;color:'+dc+';\">'+(s.dec_label||'—')+'</div>';
    h+='<div style=\"font-size:0.68rem;color:#6b7280;\">'+(s.dec_reason||'')+'</div></div></div>';
    h+='<div style=\"text-align:left;\"><div style=\"font-size:1.2rem;font-weight:800;color:'+dc+';\">'+(s.score)+'/20</div>';
    h+='<div style=\"font-size:0.6rem;color:#6b7280;\">JRF · ثقة '+s.confidence+'%</div></div></div>';
    h+='<div style=\"display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;\">';
    h+='<div style=\"background:#f8fafc;border-radius:8px;padding:8px;text-align:center;\"><div style=\"font-size:0.6rem;color:#94a3b8;\">دخول</div><div style=\"font-weight:700;\">'+s.lb+'</div></div>';
    h+='<div style=\"background:#f0fdf4;border-radius:8px;padding:8px;text-align:center;\"><div style=\"font-size:0.6rem;color:#94a3b8;\">هدف 1</div><div style=\"font-weight:700;color:#16a34a;\">'+s.t1+'</div><div style=\"font-size:0.6rem;color:#16a34a;\">+'+s.ptp+'%</div></div>';
    h+='<div style=\"background:#f0fdf4;border-radius:8px;padding:8px;text-align:center;\"><div style=\"font-size:0.6rem;color:#94a3b8;\">هدف 2</div><div style=\"font-weight:700;color:#16a34a;\">'+s.t2+'</div></div>';
    h+='<div style=\"background:#fef2f2;border-radius:8px;padding:8px;text-align:center;\"><div style=\"font-size:0.6rem;color:#94a3b8;\">وقف</div><div style=\"font-weight:700;color:#dc2626;\">'+s.sl+'</div><div style=\"font-size:0.6rem;color:#dc2626;\">-'+s.psl+'%</div></div>';
    h+='</div>';
    h+='<div style=\"display:flex;gap:6px;margin-bottom:10px;\">';
    h+='<span style=\"background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;border-radius:8px;font-size:0.7rem;\">RSI <b>'+s.rsi+'</b></span>';
    h+='<span style=\"background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;border-radius:8px;font-size:0.7rem;\">ADX <b>'+s.adx+'</b></span>';
    h+='<span style=\"background:#f8fafc;border:1px solid #e2e8f0;padding:3px 10px;border-radius:8px;font-size:0.7rem;\">R:R <b>'+s.rr+'</b></span>';
    h+='</div>';
    h+='<div style=\"display:flex;gap:8px;\">';
    h+='<a href=\"'+tvLink+'\" target=\"_blank\" style=\"background:#2563eb;color:white;padding:7px 14px;border-radius:8px;font-size:0.78rem;font-weight:700;text-decoration:none;\">📊 TradingView</a>';
    h+='<button onclick=\"closeInstant()\" style=\"background:none;border:1px solid #e2e8f0;color:#94a3b8;padding:7px 12px;border-radius:8px;font-size:0.75rem;cursor:pointer;\">✕ إغلاق</button>';
    h+='</div></div>';
    res.innerHTML=h;
  }).catch(function(){
    res.innerHTML='<div class=\"instant-wrap\"><div style=\"color:red;padding:10px;\">❌ خطأ في الاتصال</div></div>';
  });
}
// ══ JS التداول الآلي ══
var activeTradeTab = 'account';

function showTradeTab(tab) {
  activeTradeTab = tab;
  ['account','trade','positions','settings','log'].forEach(function(t) {
    var el = document.getElementById('tt-' + t);
    if (el) el.style.display = t === tab ? 'block' : 'none';
  });
  document.querySelectorAll('.tab-trade-btn').forEach(function(b, i) {
    b.classList.toggle('active', ['account','trade','positions','settings','log'][i] === tab);
  });
  if (tab === 'account') loadAccount();
  if (tab === 'positions') loadPositions();
  if (tab === 'log') loadTradeLog();
  if (tab === 'settings') loadTradingConfig();
}

function loadAccount() {
  loadStats();
  loadSchedule();
  fetch('/alpaca/account').then(function(r){return r.json();}).then(function(d) {
    if (d.ok) {
      document.getElementById('acc-equity').textContent = '$' + parseFloat(d.equity).toFixed(0);
      document.getElementById('acc-cash').textContent = '$' + parseFloat(d.cash).toFixed(0);
      document.getElementById('acc-bp').textContent = '$' + parseFloat(d.buying_power).toFixed(0);
      var pnl = parseFloat(d.pnl || 0);
      var pnlEl = document.getElementById('acc-pnl');
      pnlEl.textContent = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);
      pnlEl.style.color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
      document.getElementById('conn-dot').style.background = 'var(--green)';
      document.getElementById('conn-dot').style.boxShadow = '0 0 6px var(--green)';
      document.getElementById('conn-txt').textContent = 'متصل ✓';
      document.getElementById('conn-txt').style.color = 'var(--green)';
    } else {
      document.getElementById('conn-txt').textContent = 'غير متصل';
    }
  });
}

function loadSchedule() {
  fetch('/alpaca/schedule').then(function(r){return r.json();}).then(function(d) {
    var bar = document.getElementById('schedule-bar');
    if (bar) {
      bar.innerHTML = d.status + ' · ' + d.day + ' ' + d.ksa_time + ' KSA · 🤖 ' + d.auto_scan;
    }
  });
}

function loadStats() {
  fetch('/alpaca/stats').then(function(r){return r.json();}).then(function(d) {
    document.getElementById('stats-bar').style.display = 'block';
    document.getElementById('st-total').textContent = d.total_trades || 0;
    document.getElementById('st-wins').textContent = d.wins || 0;
    document.getElementById('st-losses').textContent = d.losses || 0;
    document.getElementById('st-wr').textContent = (d.win_rate || 0) + '%';
    var pnl = d.total_pnl || 0;
    var pnlEl = document.getElementById('st-pnl');
    pnlEl.textContent = (pnl >= 0 ? '+$' : '-$') + Math.abs(pnl).toFixed(2);
    pnlEl.style.color = pnl >= 0 ? '#16a34a' : '#dc2626';
    document.getElementById('st-today').textContent = d.today_trades || 0;
    document.getElementById('st-max').textContent = d.max_daily || 5;
  });
}

function sendOrder(side) {
  var symbol = document.getElementById('t-symbol').value.trim().toUpperCase();
  var qty = parseInt(document.getElementById('t-qty').value) || 1;
  var type = document.getElementById('t-type').value;
  var limit = document.getElementById('t-limit').value;
  var stop = document.getElementById('t-stop').value;
  if (!symbol) { showToast('أدخل الرمز'); return; }
  var data = {symbol: symbol, qty: qty, side: side, order_type: type};
  if (limit) data.limit_price = parseFloat(limit);
  if (stop) data.stop_price = parseFloat(stop);
  showToast('جاري إرسال الأمر...');
  fetch('/alpaca/trade', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)})
  .then(function(r){return r.json();}).then(function(d) {
    if (d.ok) {
      showToast((side==='buy'?'✅ تم الشراء: ':'✅ تم البيع: ') + symbol + ' × ' + qty);
    } else {
      showToast('❌ ' + (d.msg || 'فشل الأمر'));
    }
  });
}

function loadPositions() {
  fetch('/alpaca/positions').then(function(r){return r.json();}).then(function(d) {
    var list = document.getElementById('positions-list');
    if (!d.ok || !d.positions.length) {
      list.innerHTML = '<div style="color:var(--text3);font-size:0.82rem;padding:10px;">لا توجد مراكز مفتوحة</div>';
      return;
    }
    list.innerHTML = d.positions.map(function(p) {
      var pnl = parseFloat(p.pnl_pct || 0) * 100;
      var pnlCls = pnl >= 0 ? 'color:var(--green)' : 'color:var(--red)';
      var pnlSign = pnl >= 0 ? '+' : '';
      return '<div class="pos-item">' +
        '<div><div class="pos-sym">' + p.symbol + '</div>' +
        '<div class="pos-info">' + p.qty + ' سهم · دخول $' + parseFloat(p.entry).toFixed(2) + ' · الآن $' + parseFloat(p.current).toFixed(2) + '</div></div>' +
        '<div style="text-align:left;">' +
        '<div style="font-weight:700;' + pnlCls + ';">' + pnlSign + pnl.toFixed(2) + '%</div>' +
        '<div style="font-size:0.68rem;color:var(--text3);">$' + parseFloat(p.pnl).toFixed(2) + '</div>' +
        '<button class="btn-close-pos" onclick="closePos(\"' + p.symbol + '\")">✕ إغلاق</button>' +
        '</div></div>';
    }).join('');
  });
}

function closePos(symbol) {
  if (!confirm('إغلاق مركز ' + symbol + '؟')) return;
  fetch('/alpaca/close', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:symbol})})
  .then(function(r){return r.json();}).then(function(d) {
    if (d.ok) { showToast('✅ تم إغلاق ' + symbol); loadPositions(); }
    else showToast('❌ ' + (d.msg||'فشل'));
  });
}

function loadTradingConfig() {
  fetch('/alpaca/config').then(function(r){return r.json();}).then(function(d) {
    if (document.getElementById('cfg-key')) document.getElementById('cfg-key').value = d.key || '';
    if (document.getElementById('cfg-secret')) document.getElementById('cfg-secret').value = d.secret || '';
    if (document.getElementById('cfg-max')) document.getElementById('cfg-max').value = d.max_position_usd || 500;
    if (document.getElementById('cfg-loss')) document.getElementById('cfg-loss').value = d.daily_loss_limit || 2;
    if (document.getElementById('cfg-enabled')) document.getElementById('cfg-enabled').checked = d.enabled || false;
    if (document.getElementById('cfg-autobuy')) document.getElementById('cfg-autobuy').checked = d.auto_buy || false;
    if (document.getElementById('cfg-autosell')) document.getElementById('cfg-autosell').checked = d.auto_sell !== false;
  });
}

function saveTradingConfig() {
  var cfg = {
    key: document.getElementById('cfg-key').value.trim(),
    secret: document.getElementById('cfg-secret').value.trim(),
    endpoint: 'https://paper-api.alpaca.markets/v2',
    max_position_usd: parseFloat(document.getElementById('cfg-max').value) || 500,
    daily_loss_limit: parseFloat(document.getElementById('cfg-loss').value) || 2,
    enabled: document.getElementById('cfg-enabled').checked,
    auto_buy: document.getElementById('cfg-autobuy').checked,
    auto_sell: document.getElementById('cfg-autosell').checked
  };
  fetch('/alpaca/config', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(cfg)})
  .then(function(r){return r.json();}).then(function(d) {
    if (d.ok) { showToast('✅ تم حفظ الإعدادات'); loadAccount(); }
  });
}

function testConnection() {
  showToast('جاري الاختبار...');
  saveTradingConfig();
  setTimeout(loadAccount, 1000);
}

function loadTradeLog() {
  fetch('/alpaca/trades').then(function(r){return r.json();}).then(function(trades) {
    var list = document.getElementById('trade-log-list');
    if (!trades.length) {
      list.innerHTML = '<div style="color:var(--text3);font-size:0.82rem;padding:10px;">لا توجد صفقات بعد</div>';
      return;
    }
    list.innerHTML = trades.slice().reverse().map(function(t) {
      var cls = t.side === 'buy' ? 'log-buy' : 'log-sell';
      var icon = t.side === 'buy' ? '🟢' : '🔴';
      return '<div class="log-item ' + cls + '">' +
        '<span>' + icon + ' ' + t.symbol + ' × ' + t.qty + '</span>' +
        '<span>' + (t.source === 'auto_radar' ? '🤖' : '👤') + ' ' + t.time + '</span>' +
        '</div>';
    }).join('');
  });
}

// تحميل الإعدادات عند فتح الصفحة
document.addEventListener('DOMContentLoaded', function() {
  loadTradingConfig();
  setTimeout(loadAccount, 500);
});

function quickTrade(e, code, name, price, tp1, sl) {
  e.stopPropagation();
  var cfg_max = 500; // default
  var qty = Math.max(1, Math.floor(cfg_max / price));
  var msg = '⚡ تأكيد الشراء\n\n' +
    'السهم: ' + name + ' (' + code + ')\n' +
    'السعر: $' + price + '\n' +
    'الكمية: ' + qty + ' سهم\n' +
    'القيمة: $' + (qty * price).toFixed(0) + '\n' +
    'الهدف: $' + tp1 + '\n' +
    'وقف الخسارة: $' + sl + '\n\n' +
    'تأكيد؟';
  if (!confirm(msg)) return;
  fetch('/alpaca/trade', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      symbol: code, qty: qty, side: 'buy',
      order_type: 'limit', limit_price: price,
      stop_price: sl, source: 'manual_card'
    })
  }).then(function(r){return r.json();}).then(function(d) {
    if (d.ok) {
      showToast('✅ تم إرسال أمر الشراء: ' + code + ' × ' + qty);
      setTimeout(loadStats, 1000);
    } else {
      showToast('❌ ' + (d.msg || 'فشل — تحقق من إعدادات Alpaca'));
    }
  });
}

function closeInstant(){document.getElementById('instant-result').style.display='none';}

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
        <button class="add-btn-green" onclick="addStock()">➕ أضف للقائمة</button>
        <button class="add-btn-analyze" onclick="analyzeNow()">🔍 حلل الآن</button>
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-top:6px;">
        <input type="checkbox" id="noFilter" style="width:15px;height:15px;accent-color:var(--accent);cursor:pointer;">
        <label for="noFilter" style="font-size:0.75rem;color:var(--text2);cursor:pointer;">تحليل بدون فلاتر (يشمل الأسهم الصغيرة والرخيصة)</label>
      </div>
      <div style="display:none">
      </div>
      {('<div class="excl-list">مستبعدون: ' + excl_html + '</div>') if excl else ''}
    </div>'''

    # ══ قسم التداول الآلي ══
    trading_section = '''<div class="trading-wrap" id="trading-section">
  <div class="trading-header">
    <div class="trading-title">🤖 التداول الآلي — Alpaca Paper</div>
    <div class="trading-status">
      <div class="status-dot" id="conn-dot" style="background:var(--text3);"></div>
      <span id="conn-txt" style="color:var(--text3);">غير متصل</span>
    </div>
  </div>
  <div class="tabs-trading">
    <button class="tab-trade-btn active" onclick="showTradeTab('account')">💰 الحساب</button>
    <button class="tab-trade-btn" onclick="showTradeTab('trade')">📊 تداول</button>
    <button class="tab-trade-btn" onclick="showTradeTab('positions')">📁 مراكزي</button>
    <button class="tab-trade-btn" onclick="showTradeTab('settings')">⚙️ الإعدادات</button>
    <button class="tab-trade-btn" onclick="showTradeTab('log')">📋 السجل</button>
  </div>
  <!-- الحساب -->
  <div id="tt-account">
    <div class="account-grid" id="acc-grid">
      <div class="acc-card"><div class="acc-val" id="acc-equity">--</div><div class="acc-lbl">القيمة الكلية $</div></div>
      <div class="acc-card"><div class="acc-val" id="acc-cash">--</div><div class="acc-lbl">الكاش $</div></div>
      <div class="acc-card"><div class="acc-val" id="acc-bp">--</div><div class="acc-lbl">القوة الشرائية $</div></div>
      <div class="acc-card"><div class="acc-val" id="acc-pnl">--</div><div class="acc-lbl">الربح/الخسارة $</div></div>
    </div>
    <button onclick="loadAccount()" style="background:var(--accent);color:white;border:none;padding:7px 16px;border-radius:8px;font-size:0.78rem;cursor:pointer;font-family:Tajawal,sans-serif;margin-left:8px;">🔄 تحديث</button>
    <div id="schedule-bar" style="margin-top:12px;background:#eff6ff;border:1px solid #bfdbfe;border-radius:10px;padding:10px 14px;font-size:0.78rem;color:#1e40af;font-weight:600;"></div>
    <div id="stats-bar" style="margin-top:12px;background:#f8fafc;border-radius:10px;padding:10px 14px;font-size:0.75rem;color:#475569;display:none;">
      <span>📊 إجمالي: <strong id="st-total">0</strong></span> ·
      <span>✅ ربح: <strong id="st-wins" style="color:#16a34a;">0</strong></span> ·
      <span>❌ خسارة: <strong id="st-losses" style="color:#dc2626;">0</strong></span> ·
      <span>🎯 نجاح: <strong id="st-wr">0%</strong></span> ·
      <span>💰 PnL: <strong id="st-pnl">$0</strong></span> ·
      <span>📅 اليوم: <strong id="st-today">0</strong>/<span id="st-max">5</span></span>
    </div>
  </div>
  <!-- تداول -->
  <div id="tt-trade" style="display:none;">
    <div class="trade-form">
      <div class="trade-form-title">📤 إرسال أمر</div>
      <div class="trade-row">
        <input class="trade-inp" id="t-symbol" placeholder="الرمز (مثال: AAPL)">
        <input class="trade-inp" id="t-qty" type="number" placeholder="الكمية" value="1" style="max-width:90px;">
        <select class="trade-inp" id="t-type" style="max-width:120px;">
          <option value="market">Market</option>
          <option value="limit">Limit</option>
        </select>
      </div>
      <div class="trade-row">
        <input class="trade-inp" id="t-limit" type="number" placeholder="سعر Limit (اختياري)" step="0.01">
        <input class="trade-inp" id="t-stop" type="number" placeholder="Stop Loss (اختياري)" step="0.01">
      </div>
      <div style="display:flex;gap:8px;">
        <button class="btn-buy" onclick="sendOrder('buy')">🟢 شراء</button>
        <button class="btn-sell" onclick="sendOrder('sell')">🔴 بيع</button>
      </div>
    </div>
  </div>
  <!-- المراكز -->
  <div id="tt-positions" style="display:none;">
    <button onclick="loadPositions()" style="background:var(--accent);color:white;border:none;padding:7px 16px;border-radius:8px;font-size:0.78rem;cursor:pointer;font-family:Tajawal,sans-serif;margin-bottom:10px;">🔄 تحديث</button>
    <div id="positions-list"><div style="color:var(--text3);font-size:0.82rem;">اضغط تحديث</div></div>
  </div>
  <!-- الإعدادات -->
  <div id="tt-settings" style="display:none;">
    <div class="settings-grid">
      <div class="setting-row"><div class="setting-lbl">API Key</div><input class="trade-inp" id="cfg-key" type="password" placeholder="PKTPP..."></div>
      <div class="setting-row"><div class="setting-lbl">Secret Key</div><input class="trade-inp" id="cfg-secret" type="password" placeholder="24xuo..."></div>
      <div class="setting-row"><div class="setting-lbl">حجم المركز الأقصى ($)</div><input class="trade-inp" id="cfg-max" type="number" value="500"></div>
      <div class="setting-row"><div class="setting-lbl">حد الخسارة اليومي (%)</div><input class="trade-inp" id="cfg-loss" type="number" value="2" step="0.5"></div>
    </div>
    <div class="toggle-wrap"><input type="checkbox" id="cfg-enabled" style="accent-color:var(--accent);"><label class="toggle-lbl" for="cfg-enabled">تفعيل التداول الآلي</label></div>
    <div class="toggle-wrap"><input type="checkbox" id="cfg-autobuy"><label class="toggle-lbl" for="cfg-autobuy">شراء تلقائي عند إشارة BUY (ثقة 70%+)</label></div>
    <div class="toggle-wrap"><input type="checkbox" id="cfg-autosell" checked><label class="toggle-lbl" for="cfg-autosell">بيع تلقائي عند TP1 أو SL</label></div>
    <button class="save-cfg" onclick="saveTradingConfig()">💾 حفظ الإعدادات</button>
    <button onclick="testConnection()" style="background:var(--surface2);border:1.5px solid var(--border);color:var(--text2);padding:8px 20px;border-radius:8px;font-family:Tajawal,sans-serif;font-size:0.82rem;cursor:pointer;width:100%;margin-top:6px;">🔌 اختبار الاتصال</button>
  </div>
  <!-- السجل -->
  <div id="tt-log" style="display:none;">
    <button onclick="loadTradeLog()" style="background:var(--accent);color:white;border:none;padding:7px 16px;border-radius:8px;font-size:0.78rem;cursor:pointer;font-family:Tajawal,sans-serif;margin-bottom:10px;">🔄 تحديث</button>
    <div class="trade-log" id="trade-log-list"><div style="color:var(--text3);font-size:0.82rem;">اضغط تحديث</div></div>
  </div>
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

    html = HTML_TEMPLATE.replace("__SESSION__", sess).replace("__TAB_CONTENTS__", tab_contents).replace("__ADD_PANEL__", add_panel).replace("__TRADING__", trading_section)
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

@app.route("/analyze_one", methods=["POST"])
def analyze_one():
    d = request.get_json()
    code = d.get("code","").strip().upper()
    name = d.get("name","").strip() or code
    market = d.get("market","tadawul")
    period = d.get("period","daily")
    no_filter = bool(d.get("no_filter", False))
    if not code: return jsonify({"ok": False, "msg": "أدخل الرمز"})
    try:
        bdf = get_df(BENCHMARK.get(market,""), "2y", "1d")
        if market == "crypto":
            result = analyze_crypto(code, name, bdf, period, no_filter=no_filter)
        else:
            result = analyze(code, name, market, bdf, period, no_filter=no_filter)
        if not result:
            return jsonify({"ok": False, "msg": f"ما قدرت أجيب بيانات {code} — تأكد من الرمز"})
        result["medal"] = "🔍"
        result["rank_pos"] = 0
        result["is_custom"] = True
        # نحسب حكم القرار
        score = result["score"]
        conf = result["confidence"]
        adx = result["adx"]
        verdict = result["verdict"]
        bt = result["bt"]
        if verdict == "BUY" and "مشروط" not in bt:
            if conf >= 75 and score >= 15 and adx >= 25:
                result["dec_label"] = "ادخل الآن"; result["dec_icon"] = "🟢"
                result["dec_color"] = "#16a34a"; result["dec_bg"] = "#f0fdf4"
                result["dec_border"] = "#86efac"
                result["dec_reason"] = f"النقاط {score}/20 · ثقة {conf}% · اتجاه قوي"
            else:
                result["dec_label"] = "دخول جيد"; result["dec_icon"] = "🟢"
                result["dec_color"] = "#16a34a"; result["dec_bg"] = "#f0fdf4"
                result["dec_border"] = "#86efac"
                result["dec_reason"] = f"النقاط {score}/20 · ثقة {conf}%"
        elif verdict == "BUY" and "مشروط" in bt:
            result["dec_label"] = "دخول بحذر"; result["dec_icon"] = "🟡"
            result["dec_color"] = "#d97706"; result["dec_bg"] = "#fffbeb"
            result["dec_border"] = "#fcd34d"
            result["dec_reason"] = "فرصة لكن فيه تحفظ"
        elif verdict == "WAIT":
            result["dec_label"] = "انتظر"; result["dec_icon"] = "⏳"
            result["dec_color"] = "#6b7280"; result["dec_bg"] = "#f9fafb"
            result["dec_border"] = "#d1d5db"
            result["dec_reason"] = f"النقاط {score}/20 · الشروط لم تكتمل"
        else:
            result["dec_label"] = "تجنب"; result["dec_icon"] = "🔴"
            result["dec_color"] = "#dc2626"; result["dec_bg"] = "#fef2f2"
            result["dec_border"] = "#fca5a5"
            result["dec_reason"] = f"النقاط {score}/20 · الاتجاه ضعيف"
        return jsonify({"ok": True, "result": result})
    except Exception as e:
        return jsonify({"ok": False, "msg": str(e)})

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


# ══ Routes التداول الآلي ══

@app.route("/alpaca/config", methods=["GET","POST"])
def alpaca_config():
    if request.method == "POST":
        d = request.get_json()
        cfg = load_alpaca()
        cfg.update({k: d[k] for k in d if k in cfg})
        save_alpaca(cfg)
        return jsonify({"ok": True})
    return jsonify(load_alpaca())

@app.route("/alpaca/account")
def alpaca_account():
    acc = get_alpaca_account()
    if not acc or "error" in acc:
        return jsonify({"ok": False, "msg": acc.get("error","خطأ في الاتصال") if acc else "ما في إعدادات"})
    return jsonify({"ok": True,
        "equity": acc.get("equity","0"),
        "cash": acc.get("cash","0"),
        "buying_power": acc.get("buying_power","0"),
        "pnl": acc.get("unrealized_pl","0")})

@app.route("/alpaca/positions")
def alpaca_positions():
    positions = get_positions()
    if isinstance(positions, dict) and "error" in positions:
        return jsonify({"ok": False, "positions": []})
    result = []
    for p in (positions if isinstance(positions, list) else []):
        result.append({
            "symbol": p.get("symbol",""),
            "qty": p.get("qty","0"),
            "entry": p.get("avg_entry_price","0"),
            "current": p.get("current_price","0"),
            "pnl": p.get("unrealized_pl","0"),
            "pnl_pct": p.get("unrealized_plpc","0"),
            "market_value": p.get("market_value","0"),
        })
    return jsonify({"ok": True, "positions": result})

@app.route("/alpaca/orders")
def alpaca_orders():
    orders = get_orders("open")
    if isinstance(orders, dict): orders = []
    result = []
    for o in orders:
        result.append({
            "id": o.get("id",""),
            "symbol": o.get("symbol",""),
            "side": o.get("side",""),
            "qty": o.get("qty","0"),
            "type": o.get("type",""),
            "status": o.get("status",""),
            "limit_price": o.get("limit_price",""),
            "created_at": o.get("created_at","")[:10] if o.get("created_at") else "",
        })
    return jsonify({"ok": True, "orders": result})

@app.route("/alpaca/trade", methods=["POST"])
def alpaca_trade():
    d = request.get_json()
    symbol = d.get("symbol","").upper()
    side = d.get("side","buy")
    qty = d.get("qty", 1)
    order_type = d.get("order_type","market")
    limit_price = d.get("limit_price")
    stop_price = d.get("stop_price")

    if not symbol: return jsonify({"ok": False, "msg": "أدخل الرمز"})

    cfg = load_alpaca()
    if not cfg.get("key"): return jsonify({"ok": False, "msg": "أدخل API Keys أولاً"})

    result = place_order(symbol, qty, side, order_type, limit_price, stop_price)
    if not result or "error" in result:
        return jsonify({"ok": False, "msg": result.get("error","فشل الأمر") if result else "فشل"})

    # احفظ الصفقة
    save_trade({
        "symbol": symbol, "side": side, "qty": qty,
        "order_type": order_type, "limit_price": limit_price,
        "order_id": result.get("id",""),
        "status": result.get("status",""),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": d.get("source","manual")
    })
    return jsonify({"ok": True, "order_id": result.get("id",""), "status": result.get("status","")})

@app.route("/alpaca/close", methods=["POST"])
def alpaca_close():
    symbol = request.get_json().get("symbol","").upper()
    result = close_position(symbol)
    if not result or "error" in result:
        return jsonify({"ok": False, "msg": result.get("error","فشل الإغلاق") if result else "فشل"})
    return jsonify({"ok": True})

@app.route("/alpaca/trades")
def alpaca_trades():
    return jsonify(load_trades())

@app.route("/alpaca/schedule")
def alpaca_schedule():
    """حالة الجدول الزمني"""
    now = datetime.utcnow()
    weekday = now.weekday()
    hour = now.hour
    minute = now.minute
    days = ["الاثنين","الثلاثاء","الأربعاء","الخميس","الجمعة","السبت","الأحد"]
    
    # وقت السوق KSA = UTC + 3
    ksa_hour = (hour + 3) % 24
    ksa_time = f"{ksa_hour:02d}:{minute:02d}"
    
    if weekday >= 5:
        status = "🔴 السوق مغلق — عطلة"
        next_open = "الاثنين 4:30 مساءً"
    elif hour < 13 or (hour == 13 and minute < 30):
        # قبل الافتتاح
        remaining_min = (13*60+30) - (hour*60+minute)
        remaining_ksa = remaining_min
        status = f"🟡 السوق لم يفتح — يفتح خلال {remaining_min} دقيقة"
        next_open = f"اليوم 4:30 مساءً KSA"
    elif hour < 20:
        status = "🟢 السوق مفتوح الآن"
        next_open = "—"
    else:
        status = "🔴 السوق أغلق"
        next_open = "غداً 4:30 مساءً"
    
    return jsonify({
        "status": status,
        "ksa_time": ksa_time,
        "day": days[weekday],
        "next_open": next_open,
        "auto_scan": "مفعّل — يمسح تلقائياً عند 4:30 مساءً KSA"
    })

@app.route("/alpaca/stats")
def alpaca_stats():
    """إحصائيات التداول الآلي"""
    trades = load_trades()
    today = datetime.now().strftime("%Y-%m-%d")
    today_trades = [t for t in trades if t.get("time","").startswith(today)]
    buy_trades = [t for t in trades if t.get("side")=="buy" and t.get("source")=="auto_radar"]
    sell_trades = [t for t in trades if t.get("side")=="sell" and t.get("source")=="auto_radar"]

    # حساب الأرباح
    wins = [t for t in sell_trades if float(t.get("pnl",0)) > 0]
    losses = [t for t in sell_trades if float(t.get("pnl",0)) <= 0]
    total_pnl = sum(float(t.get("pnl",0)) for t in sell_trades)
    win_rate = round(len(wins)/len(sell_trades)*100, 1) if sell_trades else 0

    cfg = load_alpaca()
    max_daily = int(cfg.get("max_daily_trades", 5))

    return jsonify({
        "total_trades": len(buy_trades),
        "closed_trades": len(sell_trades),
        "wins": len(wins), "losses": len(losses),
        "win_rate": win_rate,
        "total_pnl": round(total_pnl, 2),
        "today_trades": len(today_trades),
        "remaining_today": max(0, max_daily - len([t for t in today_trades if t.get("source")=="auto_radar" and t.get("side")=="buy"])),
        "max_daily": max_daily
    })

@app.route("/alpaca/auto_trade", methods=["POST"])
def auto_trade():
    """تنفيذ صفقة تلقائية بناءً على إشارة الرادار"""
    d = request.get_json()
    signal = d.get("signal", {})
    cfg = load_alpaca()

    if not cfg.get("enabled"): return jsonify({"ok": False, "msg": "التداول الآلي معطل"})
    if not cfg.get("auto_buy"): return jsonify({"ok": False, "msg": "الشراء التلقائي معطل"})
    if signal.get("market") != "us": return jsonify({"ok": False, "msg": "فقط السوق الأمريكي"})
    if signal.get("verdict") != "BUY": return jsonify({"ok": False, "msg": "ليست إشارة شراء"})
    if signal.get("confidence", 0) < 70: return jsonify({"ok": False, "msg": "الثقة أقل من 70%"})

    # نحسب الكمية
    acc = get_alpaca_account()
    if not acc or "error" in acc: return jsonify({"ok": False, "msg": "فشل جلب الحساب"})

    max_usd = float(cfg.get("max_position_usd", 500))
    price = float(signal.get("price", 0))
    if price <= 0: return jsonify({"ok": False, "msg": "سعر غير صالح"})

    qty = max(1, int(max_usd / price))
    limit_price = signal.get("lb")  # سعر الدخول المقترح

    result = place_order(signal["code"], qty, "buy", "limit", limit_price)
    if not result or "error" in result:
        return jsonify({"ok": False, "msg": result.get("error","فشل") if result else "فشل"})

    save_trade({
        "symbol": signal["code"], "name": signal.get("name",""),
        "side": "buy", "qty": qty, "order_type": "limit",
        "limit_price": limit_price, "tp1": signal.get("t1"),
        "sl": signal.get("sl"), "order_id": result.get("id",""),
        "status": result.get("status",""), "score": signal.get("score",0),
        "confidence": signal.get("confidence",0),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "auto_radar"
    })
    return jsonify({"ok": True, "qty": qty, "order_id": result.get("id","")})


# نشغّل المراقب على مستوى الموديول عشان يشتغل سواء بـ python أو gunicorn
try:
    start_monitor()
except Exception as _e:
    print(f"تحذير: مراقب الصفقات لم يبدأ: {_e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  ⚡ جلال رادار v5.0 — Clean & Smart")
    print(f"  تاسي: {len(DEFAULT_TADAWUL)} سهم")
    print(f"  أمريكي: {len(DEFAULT_US)} سهم")
    print(f"  عملات: {len(DEFAULT_CRYPTO)} عملة")
    print("  🤖 مراقب الصفقات شغّال")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
