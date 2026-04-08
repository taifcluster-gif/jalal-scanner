#!/usr/bin/env python3
# ══════════════════════════════════════════════════
# جلال رادار v4.1 — Halal Professional Edition
# ══════════════════════════════════════════════════
from flask import Flask, jsonify, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings, threading, webbrowser, os, json
warnings.filterwarnings("ignore")

app = Flask(__name__)

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

# ══ قوائم الأسهم الشرعية ══
DEFAULT_TADAWUL = {
    "4326":"الماجدية","4220":"إعمار","9527":"ألف ميم ياء","9599":"طاقات",
    "9560":"وجا","2080":"الغاز","4150":"التعمير","3050":"أسمنت الجنوب",
    "3090":"أسمنت تبوك","3020":"أسمنت اليمامة","7200":"إم آي إس",
    "4009":"السعودي الألماني","4210":"الأبحاث والإعلام","2110":"الكابلات السعودية",
    "2060":"التصنيع","2300":"صناعة الورق","4250":"جبل عمر","3030":"أسمنت السعودية",
    "2090":"جبسكو","2130":"صدق","3003":"أسمنت المدينة","4165":"الماجد للعود",
    "6002":"هرفي للأغذية","2160":"أميانتيت","3010":"أسمنت العربية",
    "3092":"أسمنت الرياض","3080":"أسمنت الشرقية","1202":"مبكو","4020":"العقارية",
    "3007":"الواحة","2083":"مرافق","4019":"اس ام سي للرعاية","2310":"سبكيم",
    "1183":"سهل","3005":"أسمنت ام القرى","4162":"المنجم","2270":"سدافكو",
    "4090":"طيبة","3060":"أسمنت ينبع","4260":"بدجت السعودية","4192":"السيف غاليري",
    "2082":"أكوا","4031":"الخدمات الأرضية","4017":"فقيه الطبية","2282":"نقي",
    "4291":"الوطنية للتعليم","2223":"لوبريف","2330":"المتقدمة","4163":"الدواء",
    "4071":"العربية","2050":"مجموعة صافولا","1810":"سيرا","4018":"الموسى",
    "7040":"قو للإتصالات","4002":"المواساة","7202":"سلوشنز","2100":"وفرة",
    "1303":"الصناعات الكهربائية","2240":"صناعات","4321":"سينومي سنترز",
    "2010":"سابك","4130":"درب السعودية","2320":"البابطين","4333":"تعليم ريت",
    "2350":"كيان السعودية","2030":"المصافي","6050":"الأسماك","4191":"أبو معطي",
    "4290":"الخليج للتدريب","2040":"الخزف السعودي","4270":"طباعة وتغليف",
    "4323":"سمو","2286":"المطاحن الرابعة","1304":"اليمامة للحديد",
    "6004":"كاتريون","6016":"برغرايززر","2250":"المجموعة السعودية",
    "4080":"سناد القابضة","6090":"جازادكو","2120":"متطورة","2140":"أيان",
    "4040":"سابتكو","4050":"ساسكو","4263":"سال","2381":"الحفر العربية",
    "4030":"البحري","4011":"لازوردي","2280":"المراعي","4180":"مجموعة فتيحي",
    "2285":"المطاحن العربية","1320":"أنابيب السعودية","6070":"الجوف",
    "4160":"ثمار","2200":"أنابيب","1302":"بوان","2340":"ارتيكس",
    "4300":"دار الأركان","4001":"سينومي ريتيل","7020":"إتحاد إتصالات",
    "2281":"تنمية","4084":"دراية","4003":"إكسترا","4164":"النهدي",
    "6040":"تبوك الزراعية","2190":"سيسكو القابضة","2070":"الدوائية",
    "4007":"الحمادي","4240":"سينومي ريتيل","2170":"اللجين",
    "4015":"جمجوم فارما","2290":"ينساب","4194":"محطة البناء",
    "2222":"أرامكو السعودية","4310":"مدينة المعرفة","3091":"أسمنت الجوف",
    "1111":"مجموعة تداول","4190":"جرير","1211":"معادن","2370":"مسك",
    "2020":"سابك للمغذيات","4322":"رتال","4232":"مدى","7010":"STC",
    "1010":"الرياض","1080":"الأهلي","1120":"الراجحي","1050":"الإنماء",
    "5110":"الكهرباء","7030":"زين","4072":"MBC","4325":"مسار",
    "4144":"رؤوم","4005":"رعاية","1831":"مهارة","1833":"الموارد",
    "1321":"أنابيب الشرق","4100":"مكة","2180":"فيبكو","3008":"الكثيري",
    "2230":"الكيميائية","4013":"سليمان الحبيب",
}

DEFAULT_US = {
    # ══ تكنولوجيا ══
    "AAPL":"Apple","MSFT":"Microsoft","GOOGL":"Alphabet A","GOOG":"Alphabet C",
    "META":"Meta Platforms","NVDA":"NVIDIA","AMD":"AMD","INTC":"Intel",
    "QCOM":"Qualcomm","AVGO":"Broadcom","TXN":"Texas Instruments","MU":"Micron",
    "AMAT":"Applied Materials","LRCX":"Lam Research","KLAC":"KLA Corp",
    "MRVL":"Marvell Technology","MPWR":"Monolithic Power","ENPH":"Enphase Energy",
    "ON":"ON Semiconductor","SWKS":"Skyworks Solutions","QRVO":"Qorvo",
    "MCHP":"Microchip Technology","ADI":"Analog Devices","NXPI":"NXP Semiconductors",
    "WOLF":"Wolfspeed","AMBA":"Ambarella","ALGM":"Allegro MicroSystems",
    "CRUS":"Cirrus Logic","DIOD":"Diodes Inc","FORM":"FormFactor",
    "IPGP":"IPG Photonics","ACLS":"Axcelis Technologies","ONTO":"Onto Innovation",
    "NOVT":"Novanta","COHU":"Cohu","ICHR":"Ichor Holdings",
    "UCTT":"Ultra Clean Holdings","MKSI":"MKS Instruments",
    # ══ برمجيات ══
    "CRM":"Salesforce","NOW":"ServiceNow","SNOW":"Snowflake","DDOG":"Datadog",
    "HUBS":"HubSpot","WDAY":"Workday","VEEV":"Veeva Systems","ANSS":"ANSYS",
    "CDNS":"Cadence Design","SNPS":"Synopsys","PTC":"PTC Inc","ADSK":"Autodesk",
    "ORCL":"Oracle","SAP":"SAP SE","INTU":"Intuit","PCTY":"Paylocity",
    "PAYC":"Paycom","SMAR":"Smartsheet","ZI":"ZoomInfo","GTLB":"GitLab",
    "ESTC":"Elastic","MDB":"MongoDB","DSGX":"Descartes Systems","APPF":"AppFolio",
    "NCNO":"nCino","JAMF":"Jamf Holding","BRZE":"Braze","ALTR":"Altair Engineering",
    "AZPN":"Aspen Technology","EVBG":"Everbridge","ALRM":"Alarm.com",
    "FIVN":"Five9","NICE":"NICE Systems","MANH":"Manhattan Associates",
    "PRGS":"Progress Software","SPSC":"SPS Commerce","BLKB":"Blackbaud",
    "PCOR":"Procore Technologies","TOST":"Toast Inc","FOUR":"Shift4 Payments",
    "PEGA":"Pegasystems","EGHT":"8x8 Inc","AMSWA":"American Software",
    # ══ أمن سيبراني ══
    "ZS":"Zscaler","CRWD":"CrowdStrike","PANW":"Palo Alto Networks",
    "FTNT":"Fortinet","NET":"Cloudflare","S":"SentinelOne","OKTA":"Okta",
    "TENB":"Tenable","RPD":"Rapid7","VRNS":"Varonis Systems","QLYS":"Qualys",
    "CYBR":"CyberArk","ACIW":"ACI Worldwide","MIME":"Mimecast",
    "VIAV":"Viavi Solutions","OSPN":"OneSpan","CWAN":"Clearwater Analytics",
    # ══ سحابة وبنية تحتية ══
    "AMZN":"Amazon","IBM":"IBM","HPQ":"HP Inc","HPE":"Hewlett Packard Enterprise",
    "DELL":"Dell Technologies","NTAP":"NetApp","PSTG":"Pure Storage",
    "AKAM":"Akamai","FSLY":"Fastly","BAND":"Bandwidth Inc",
    # ══ صحة ودواء ══
    "JNJ":"Johnson & Johnson","UNH":"UnitedHealth","ABBV":"AbbVie",
    "TMO":"Thermo Fisher","ABT":"Abbott Labs","DHR":"Danaher",
    "ISRG":"Intuitive Surgical","SYK":"Stryker","BSX":"Boston Scientific",
    "EW":"Edwards Lifesciences","BDX":"Becton Dickinson","ZBH":"Zimmer Biomet",
    "HOLX":"Hologic","DXCM":"DexCom","ALGN":"Align Technology",
    "PODD":"Insulet Corp","INSP":"Inspire Medical","NVCR":"NovaCure",
    "GMED":"Globus Medical","NVST":"Envista Holdings","OMCL":"Omnicell",
    "AMED":"Amedisys","VRTX":"Vertex Pharma","REGN":"Regeneron",
    "ALNY":"Alnylam Pharma","BEAM":"Beam Therapeutics","CRSP":"CRISPR Therapeutics",
    "NTLA":"Intellia Therapeutics","RXRX":"Recursion Pharma","TGTX":"TG Therapeutics",
    "LNTH":"Lantheus Holdings","HALO":"Halozyme","ARWR":"Arrowhead Pharma",
    "MDGL":"Madrigal Pharma","PAHC":"Phibro Animal Health","PCRX":"Pacira BioSciences",
    "PRGO":"Perrigo","SLP":"Simulations Plus","USPH":"US Physical Therapy",
    "ACHC":"Acadia Healthcare","ADUS":"Addus HomeCare","AMPH":"Amphastar Pharma",
    "CORT":"Corcept Therapeutics","DOCS":"Doximity","XRAY":"Dentsply Sirona",
    "NKTR":"Nektar Therapeutics","GEHC":"GE HealthCare","MMSI":"Merit Medical",
    "IRTC":"iRhythm Technologies","SWAV":"ShockWave Medical","AXNX":"Axonics",
    "ATEC":"Alphatec Holdings","NVRO":"Nevro Corp","ANGO":"AngioDynamics",
    "LMAT":"LeMaitre Vascular","NARI":"Inari Medical","PRCT":"PROCEPT BioRobotics",
    "SILK":"Silk Road Medical","ILMN":"Illumina","NTRA":"Natera","EXAS":"Exact Sciences",
    "MRNA":"Moderna","BNTX":"BioNTech","MCK":"McKesson Corp",
    "CAH":"Cardinal Health","HCA":"HCA Healthcare","ENSG":"Ensign Group",
    "PNTG":"Pennant Group","GDRX":"GoodRx","TDOC":"Teladoc Health",
    "ONEM":"One Medical","OPRX":"OptimizeRx","HCAT":"Health Catalyst",
    # ══ صناعة ══
    "HON":"Honeywell","MMM":"3M","GE":"GE Aerospace","RTX":"Raytheon Technologies",
    "LMT":"Lockheed Martin","NOC":"Northrop Grumman","BA":"Boeing",
    "CAT":"Caterpillar","DE":"John Deere","EMR":"Emerson Electric",
    "ROK":"Rockwell Automation","ITW":"Illinois Tool Works","ETN":"Eaton",
    "PH":"Parker Hannifin","DOV":"Dover Corp","FTV":"Fortive","AME":"AMETEK",
    "VRSK":"Verisk Analytics","IEX":"IDEX Corp","RBC":"RBC Bearings",
    "GGG":"Graco","MIDD":"Middleby Corp","NDSN":"Nordson","TTC":"Toro Company",
    "AAON":"AAON Inc","HLIO":"Helios Technologies","MYRG":"MYR Group",
    "ROAD":"Construction Partners","SITE":"SiteOne Landscape","TREX":"Trex Company",
    "ACA":"Arcosa","ATKR":"Atkore","BCC":"Boise Cascade","BLDR":"Builders FirstSource",
    "IBP":"Installed Building Products","PATK":"Patrick Industries",
    "SSD":"Simpson Manufacturing","STLD":"Steel Dynamics","CMC":"Commercial Metals",
    "NUE":"Nucor","RS":"Reliance Steel","ATI":"ATI Inc","HWM":"Howmet Aerospace",
    "KALU":"Kaiser Aluminum","CRS":"Carpenter Technology","WIRE":"Encore Wire",
    "IIIN":"Insteel Industries",
    # ══ طاقة متجددة ══
    "FSLR":"First Solar","NEE":"NextEra Energy","SEDG":"SolarEdge",
    "RUN":"Sunrun","NOVA":"Sunnova Energy","STEM":"Stem Inc","BE":"Bloom Energy",
    "PLUG":"Plug Power","ARRY":"Array Technologies","CWEN":"Clearway Energy",
    "ORA":"Ormat Technologies","AES":"AES Corp","CLNE":"Clean Energy Fuels",
    "AMRC":"Ameresco","HASI":"Hannon Armstrong","SPWR":"SunPower",
    "MAXN":"Maxeon Solar","CSIQ":"Canadian Solar","JKS":"JinkoSolar","DQ":"Daqo New Energy",
    # ══ طاقة تقليدية ══
    "XOM":"ExxonMobil","CVX":"Chevron","COP":"ConocoPhillips","EOG":"EOG Resources",
    "OXY":"Occidental Petroleum","PSX":"Phillips 66","VLO":"Valero Energy",
    "MPC":"Marathon Petroleum","HES":"Hess Corp","DVN":"Devon Energy",
    "FANG":"Diamondback Energy","APA":"APA Corp","MRO":"Marathon Oil",
    "CTRA":"Coterra Energy","SM":"SM Energy","MTDR":"Matador Resources",
    "CHRD":"Chord Energy","NOG":"Northern Oil and Gas","SLB":"Schlumberger",
    "HAL":"Halliburton","BKR":"Baker Hughes","FTI":"TechnipFMC",
    "HP":"Helmerich & Payne","TRGP":"Targa Resources","OKE":"ONEOK","WMB":"Williams Companies",
    # ══ كيماويات ══
    "LIN":"Linde","APD":"Air Products","ECL":"Ecolab","SHW":"Sherwin-Williams",
    "PPG":"PPG Industries","RPM":"RPM International","ALB":"Albemarle",
    "FMC":"FMC Corp","CE":"Celanese","HUN":"Huntsman Corp","CC":"Chemours",
    "TROX":"Tronox Holdings","KWR":"Quaker Houghton","CBT":"Cabot Corp",
    "AVNT":"Avient Corp","HWKN":"Hawkins Inc","BCPC":"Balchem Corp",
    # ══ استهلاك ══
    "COST":"Costco","HD":"Home Depot","NKE":"Nike","SBUX":"Starbucks",
    "MCD":"McDonald's","YUM":"Yum! Brands","CMG":"Chipotle","DPZ":"Domino's Pizza",
    "WMT":"Walmart","TGT":"Target","ULTA":"Ulta Beauty","LULU":"Lululemon",
    "ROST":"Ross Stores","TJX":"TJX Companies","BBY":"Best Buy","LOW":"Lowe's",
    "WSM":"Williams-Sonoma","RH":"Restoration Hardware","ETSY":"Etsy",
    "CHWY":"Chewy","BURL":"Burlington Stores","FIVE":"Five Below",
    "TXRH":"Texas Roadhouse","WING":"Wingstop","SHAK":"Shake Shack",
    "CALM":"Cal-Maine Foods","JJSF":"J&J Snack Foods","LANC":"Lancaster Colony",
    "USFD":"US Foods","SYY":"Sysco","PFGC":"Performance Food Group",
    "CASY":"Casey's General","MUSA":"Murphy USA","PTLO":"Portillo's",
    "BROS":"Dutch Bros Coffee","HAIN":"Hain Celestial","SMPL":"Simply Good Foods",
    "PZZA":"Papa John's","CAKE":"Cheesecake Factory","EAT":"Brinker International",
    "DINE":"Dine Brands","DNUT":"Krispy Kreme","NDLS":"Noodles & Company",
    # ══ مواصلات ولوجستيك ══
    "UBER":"Uber","LYFT":"Lyft","ABNB":"Airbnb","BKNG":"Booking Holdings",
    "EXPE":"Expedia","TCOM":"Trip.com Group","UPS":"United Parcel Service",
    "FDX":"FedEx","XPO":"XPO Logistics","CHRW":"CH Robinson",
    "EXPD":"Expeditors International","FWRD":"Forward Air","HUBG":"Hub Group",
    "JBHT":"JB Hunt Transport","ODFL":"Old Dominion Freight","WERN":"Werner Enterprises",
    "ARCB":"ArcBest Corp","GXO":"GXO Logistics","RXO":"RXO Inc",
    "TFII":"TFI International","LSTR":"Landstar System","R":"Ryder System",
    "AL":"Air Lease","ATSG":"Air Transport Services","MNRO":"Monro",
    # ══ ترفيه وإعلام ══
    "NFLX":"Netflix","IMAX":"IMAX Corp","CNK":"Cinemark","SPOT":"Spotify",
    "LYV":"Live Nation","MSGS":"Madison Square Garden","EDR":"Endeavor Group",
    "TTWO":"Take-Two Interactive","EA":"Electronic Arts","RBLX":"Roblox",
    "U":"Unity Software","SEAT":"Vivid Seats",
    # ══ تقنية مالية ══
    "PYPL":"PayPal","SQ":"Block Inc","AFRM":"Affirm","UPST":"Upstart",
    "SOFI":"SoFi Technologies","OPFI":"OppFi","EVTC":"EVERTEC",
    "FLYW":"Flywire","RELY":"Remitly Global","BILL":"Bill.com",
    # ══ عقارات وبناء ══
    "LEN":"Lennar Corp","DHI":"D.R. Horton","PHM":"PulteGroup","TOL":"Toll Brothers",
    "NVR":"NVR Inc","MDC":"MDC Holdings","TMHC":"Taylor Morrison","MHO":"M/I Homes",
    "SKY":"Skyline Champion","CVCO":"Cavco Industries","WSO":"Watsco",
    "ROCK":"Gibraltar Industries","BECN":"Beacon Roofing","GMS":"GMS Inc",
    "AWI":"Armstrong World","APOG":"Apogee Enterprises","DOOR":"Masonite International",
    "JELD":"JELD-WEN","MAS":"Masco Corp","OC":"Owens Corning",
    "FBHS":"Fortune Brands","HXL":"Hexcel","AZEK":"AZEK Company","PGTI":"PGT Innovations",
    # ══ زراعة وغذاء ══
    "ADM":"Archer-Daniels-Midland","BG":"Bunge","TSN":"Tyson Foods",
    "CAG":"Conagra Brands","SJM":"JM Smucker","MKC":"McCormick",
    "CPB":"Campbell Soup","K":"Kellogg","GIS":"General Mills","HRL":"Hormel Foods",
    "MDLZ":"Mondelez International","HSY":"Hershey","TR":"Tootsie Roll",
    "JBSS":"John B. Sanfilippo","UNFI":"United Natural Foods",
    "SPTN":"SpartanNash","CVGW":"Calavo Growers","CHEF":"Chefs' Warehouse",
    # ══ بيئة ══
    "WM":"Waste Management","RSG":"Republic Services","CLH":"Clean Harbors",
    "HCCI":"Heritage Crystal Clean","SRCL":"Stericycle","ERII":"Energy Recovery",
    # ══ تعليم ══
    "CHGG":"Chegg","PRDO":"Perdoceo Education","STRA":"Strategic Education",
    "GHC":"Graham Holdings","LOPE":"Grand Canyon Education","TWOU":"2U Inc",
    "COUR":"Coursera","DUOL":"Duolingo",
    # ══ ذكاء اصطناعي وروبوتيكا ══
    "TSLA":"Tesla","PATH":"UiPath","AI":"C3.ai","SYM":"Symbotic",
    "KRNT":"Kornit Digital","XMTR":"Xometry","VERI":"Veritone",
    "ASAN":"Asana","MNDY":"Monday.com","APP":"AppLovin","IRBT":"iRobot",
    "LAZR":"Luminar Technologies","MVIS":"MicroVision","INVZ":"Innoviz Technologies",
    # ══ خدمات متنوعة ══
    "CTAS":"Cintas Corp","ROL":"Rollins Inc","ABM":"ABM Industries",
    "BFAM":"Bright Horizons","KELYA":"Kelly Services","KFY":"Korn Ferry",
    "MAN":"ManpowerGroup","ASGN":"ASGN Inc","CDW":"CDW Corp",
    "NSIT":"Insight Direct","GPN":"Global Payments","WEX":"WEX Inc",
    "PRLB":"Proto Labs","XPEL":"XPEL Inc","HNI":"HNI Corp",
    "MATX":"Matson Inc","HURN":"Huron Consulting","ICF":"ICF International",
    "MGRC":"McGrath RentCorp","TRN":"Trinity Industries",
}

DEFAULT_CRYPTO = {
    "BTCUSDT":"Bitcoin","ETHUSDT":"Ethereum","BNBUSDT":"BNB",
    "SOLUSDT":"Solana","XRPUSDT":"Ripple","ADAUSDT":"Cardano",
    "DOTUSDT":"Polkadot","LINKUSDT":"Chainlink","AVAXUSDT":"Avalanche",
    "MATICUSDT":"Polygon","ATOMUSDT":"Cosmos","UNIUSDT":"Uniswap",
    "LTCUSDT":"Litecoin","XLMUSDT":"Stellar","ALGOUSDT":"Algorand",
    "VETUSDT":"VeChain","FILUSDT":"Filecoin","AAVEUSDT":"Aave",
    "SANDUSDT":"Sandbox","MANAUSDT":"Decentraland",
    "NEARUSDT":"NEAR Protocol","FTMUSDT":"Fantom","APTUSDT":"Aptos",
    "ARBUSDT":"Arbitrum","OPUSDT":"Optimism","INJUSDT":"Injective",
    "SUIUSDT":"Sui","SEIUSDT":"Sei","TIAUSDT":"Celestia",
    "PYTHUSDT":"Pyth Network","JUPUSDT":"Jupiter","WIFUSDT":"dogwifhat",
    "BONKUSDT":"Bonk","PEPEUSDT":"Pepe","SHIBUSDT":"Shiba Inu",
    "DOGEUSDT":"Dogecoin","TRXUSDT":"Tron","TONUSDT":"Toncoin",
    "KASUSDT":"Kaspa","FETUSDT":"Fetch.AI","RENDERUSDT":"Render",
}

SECTORS_TADAWUL = {
    "البتروكيماويات ⚗️": ["2010","2020","2060","2070","2080","2090","2100","2110","2150","2170","2180","2220","2223","2290","2350","2001"],
    "الطاقة ⚡":         ["2222","5110","5120","2082","2083","2084"],
    "الاتصالات 📡":      ["7010","7020","7030","7040","7200","7201","7202","7203","7204","7211"],
    "التجزئة 🛒":        ["4190","4192","4200","4210","4220","4230","4240","4260","4270","4001","4003"],
    "الأسمنت 🏗️":       ["3003","3005","3007","3008","3010","3020","3030","3040","3050","3060","3080","3090","3091","3092"],
    "الصناعة 🔧":        ["2140","2160","2200","2210","2280","2285","2300","2310","2320","2330","4030","4040","4072"],
    "العقارات 🏢":       ["4020","4090","4100","4110","4130","4150","4160","4220","4250","4300","4310","4322"],
    "الصحة 🏥":          ["4002","4004","4005","4007","4009","4013","4015","4017","4018","4019","4021","4163","4164"],
    "التقنية 💻":        ["7202","1111","4084","4083","4082","1183","1831"],
    "الغذاء 🍽️":        ["2050","2280","2282","2283","2285","2286","2287","6001","6002","6004","6010","6012","6013","6014","6016","6017"],
}
SECTORS_US = {
    "تكنولوجيا 💻":     ["AAPL","MSFT","GOOGL","META","NVDA","AMD","QCOM","INTC","AMAT","LRCX","KLAC","MRVL","AVGO","TXN","MU","CRM","NOW","SNOW","DDOG","ADSK","ORCL"],
    "أمن سيبراني 🔒":   ["ZS","CRWD","PANW","FTNT","NET","S","OKTA","CYBR","TENB"],
    "طاقة متجددة ☀️":   ["ENPH","FSLR","NEE","SEDG","RUN","BE","ARRY","CSIQ","JKS"],
    "صحة 🏥":           ["JNJ","UNH","ABBV","TMO","ABT","DHR","ISRG","SYK","VRTX","REGN","ILMN"],
    "استهلاك 🛒":       ["COST","HD","NKE","SBUX","MCD","WMT","TGT","ULTA","LULU","AMZN"],
    "طاقة ⛽":          ["XOM","CVX","COP","EOG","OXY","SLB","HAL"],
    "كيماويات 🧪":      ["LIN","APD","ECL","SHW","ALB"],
    "صناعة 🔧":         ["HON","GE","CAT","DE","EMR","ROK","ETN","BA","LMT"],
    "مواصلات 🚚":       ["UBER","LYFT","ABNB","BKNG","UPS","FDX","ODFL"],
    "ذكاء اصطناعي 🤖":  ["TSLA","PATH","AI","SYM","APP","ASAN","MNDY"],
}
SECTORS_CRYPTO = {
    "Layer 1 ⛓️":      ["BTCUSDT","ETHUSDT","SOLUSDT","ADAUSDT","AVAXUSDT","NEARUSDT","APTUSDT","SUIUSDT","SEIUSDT"],
    "DeFi 🏦":         ["UNIUSDT","AAVEUSDT","INJUSDT"],
    "Web3 🌐":         ["LINKUSDT","FILUSDT","RENDERUSDT","FETUSDT"],
    "Layer 2 ⚡":      ["MATICUSDT","ARBUSDT","OPUSDT"],
    "الميتافيرس 🎮":   ["SANDUSDT","MANAUSDT"],
    "خدمات 🔧":        ["BNBUSDT","XRPUSDT","LTCUSDT","XLMUSDT","TRXUSDT","TONUSDT","DOTUSDT","ATOMUSDT","ALGOUSDT","VETUSDT","KASUSDT"],
    "ميم 🐸":          ["DOGEUSDT","SHIBUSDT","PEPEUSDT","BONKUSDT","WIFUSDT"],
    "أوراكل 🔮":        ["PYTHUSDT","JUPUSDT","TIAUSDT","FTMUSDT"],
}
SECTORS_MAP = {"tadawul":SECTORS_TADAWUL,"us":SECTORS_US,"crypto":SECTORS_CRYPTO}
BENCHMARK = {"tadawul":"^TASI.SR","us":"^GSPC","crypto":"BTC-USD"}

scan_state = {
    "tadawul":{"data":None,"last_scan":None,"status":"idle","progress":0,"total":0},
    "us":     {"data":None,"last_scan":None,"status":"idle","progress":0,"total":0},
    "crypto": {"data":None,"last_scan":None,"status":"idle","progress":0,"total":0},
}

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
    return str(max(1,days-1))+"-"+str(days+2)+" أيام"

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

def get_sector_heat(data,market="tadawul"):
    sectors=SECTORS_MAP.get(market,SECTORS_TADAWUL)
    res={}
    for sector,codes in sectors.items():
        scores=[s["score"] for s in data if s["code"] in codes]
        if scores:
            avg=round(sum(scores)/len(scores),1)
            buy=sum(1 for s in data if s["code"] in codes and s["verdict"]=="BUY")
            res[sector]={"avg":avg,"count":len(scores),"buy":buy}
    return dict(sorted(res.items(),key=lambda x:-x[1]["avg"]))

def backtest_stock(code, market="tadawul", lookback_days=180):
    try:
        ticker = code+".SR" if market=="tadawul" else code
        df = get_df(ticker, "2y", "1d")
        if df.empty or len(df) < lookback_days+50: return None
        trades = []
        for i in range(50, len(df)-lookback_days, 5):
            window = df.iloc[:i+lookback_days].copy()
            if len(window) < 50: continue
            price = float(window["Close"].iloc[-1])
            e20  = float(ema(window["Close"],20).iloc[-1])
            e50  = float(ema(window["Close"],50).iloc[-1])
            e200 = float(ema(window["Close"],200).iloc[-1])
            rv   = float(rsi_f(window["Close"]).iloc[-1])
            ml,sg,_ = macd_f(window["Close"])
            adv  = float(adx_f(window["High"],window["Low"],window["Close"]).iloc[-1])
            av   = float(atr_f(window["High"],window["Low"],window["Close"]).iloc[-1])
            va   = float(sma(window["Volume"],20).iloc[-1])
            sk,sdv = stoch_f(window["High"],window["Low"],window["Close"])
            c1=price>e20; c2=e20>e50; c3=price>e200
            c4=float(ml.iloc[-1])>float(sg.iloc[-1])
            c6=40<=rv<=70; c8=adv>20
            c9=float(sk.iloc[-1])>20
            c10=float(window["Volume"].iloc[-1])>va*1.2
            score=(2 if c1 else 0)+(2 if c2 else 0)+(2 if c3 else 0)+(2 if c4 else 0)+\
                  (2 if c6 else 0)+(2 if c8 else 0)+(2 if c9 else 0)+(2 if c10 else 0)
            if score >= 13 and av > 0:
                sl_price = price - av*1.5
                tp_price = price + av*2.0
                rr = (tp_price-price)/max(price-sl_price,0.001)
                if rr < 1.0: continue
                future = df.iloc[i+lookback_days:i+lookback_days+20]
                if len(future) < 5: continue
                hit_tp = any(future["High"] >= tp_price)
                hit_sl = any(future["Low"] <= sl_price)
                if hit_tp and hit_sl:
                    tp_idx = next((j for j,h in enumerate(future["High"]) if h>=tp_price), 999)
                    sl_idx = next((j for j,l in enumerate(future["Low"]) if l<=sl_price), 999)
                    result = "WIN" if tp_idx < sl_idx else "LOSS"
                elif hit_tp: result = "WIN"
                elif hit_sl: result = "LOSS"
                else:
                    last_price = float(future["Close"].iloc[-1])
                    result = "WIN" if last_price > price else "LOSS"
                pnl = round((tp_price-price)/price*100,2) if result=="WIN" else round((sl_price-price)/price*100,2)
                trades.append({"result":result,"pnl":pnl,"entry":round(price,3),"tp":round(tp_price,3),"sl":round(sl_price,3),"score":score})
        if not trades: return None
        wins = [t for t in trades if t["result"]=="WIN"]
        losses = [t for t in trades if t["result"]=="LOSS"]
        win_rate = round(len(wins)/len(trades)*100,1)
        avg_win = round(sum(t["pnl"] for t in wins)/len(wins),2) if wins else 0
        avg_loss = round(sum(t["pnl"] for t in losses)/len(losses),2) if losses else 0
        total_pnl = round(sum(t["pnl"] for t in trades),2)
        expectancy = round((win_rate/100*avg_win) + ((1-win_rate/100)*avg_loss),2)
        return {
            "code":code,"trades":len(trades),"win_rate":win_rate,
            "avg_win":avg_win,"avg_loss":avg_loss,"total_pnl":total_pnl,
            "expectancy":expectancy,"wins":len(wins),"losses":len(losses),
            "grade": "A" if win_rate>=65 else ("B" if win_rate>=55 else ("C" if win_rate>=45 else "D"))
        }
    except: return None

backtest_cache = {}

def run_backtest(market):
    custom = load_custom()
    if market == "tadawul": stocks = {**DEFAULT_TADAWUL, **custom.get("tadawul",{})}
    elif market == "us": stocks = {**DEFAULT_US, **custom.get("us",{})}
    else: return []
    results = []
    sample = list(stocks.items())[:15]
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(backtest_stock, code, market): (code,name) for code,name in sample}
        for f in as_completed(futs):
            r = f.result()
            if r:
                code,name = futs[f]
                r["name"] = name
                results.append(r)
    results.sort(key=lambda x:-x["win_rate"])
    backtest_cache[market] = {"data":results,"ts":datetime.now().strftime("%Y-%m-%d %H:%M")}
    return results

def calc_rank(s):
    r=0; reasons=[]
    jrf=s.get("score",0); r+=jrf*2
    if jrf>=18: reasons.append("JRF ممتاز "+str(jrf)+"/20 🏆")
    elif jrf>=15: reasons.append("JRF قوي "+str(jrf)+"/20 ✅")
    stars=s.get("stars",0); r+=stars*8
    if stars==3: reasons.append("فوق المتوسطات الثلاثة ⭐⭐⭐")
    elif stars==2: reasons.append("يومي + أسبوعي ⭐⭐")
    rsv=s.get("rsv",0)
    if rsv>5: r+=15; reasons.append("أقوى من السوق بكثير 📈")
    elif rsv>2: r+=10; reasons.append("أقوى من السوق 📈")
    elif rsv>0: r+=5
    adx=s.get("adx",0)
    if adx>40: r+=10; reasons.append("ترند قوي جداً ADX "+str(adx)+" 🔥")
    elif adx>25: r+=6; reasons.append("ترند جيد ADX "+str(adx))
    elif adx>20: r+=3
    liq=s.get("liq",0)
    if liq>100: r+=5; reasons.append("سيولة عالية 💧")
    elif liq>10: r+=2
    if s.get("exp"): r+=8; reasons.append("حجم شاذ 🚀 ×"+str(s.get("vr",1)))
    rr=s.get("rr",0)
    if rr>=2.5: r+=5; reasons.append("R:R ممتاز "+str(rr))
    elif rr>=1.5: r+=3
    if "مشروط" in s.get("bt",""): r-=5; reasons.append("⚠️ BUY مشروط")
    s["rank_score"]=round(r,1); s["rank_reasons"]=reasons
    return r

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
        # ══ عرض السيولة بالوحدة الصحيحة ══
        liq_unit = "م.ر" if market=="tadawul" else "م.$"
        liq_lbl=("عالية 🟢 "+str(liq)+" "+liq_unit) if liq>100 else (("متوسطة 🟡 "+str(liq)+" "+liq_unit) if liq>10 else ("منخفضة 🔴 "+str(liq)+" "+liq_unit))
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
        tp=round((atr_v*1.5/p)*100,2); tg=round(atr_v*0.3,3); be=round(p+(p-sl)*0.1,3)
        if score>=15 and rr>=1.3: v,pr,bt,note="BUY",1,"🟢 BUY",""
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
            {"n":"MACD Rising","ok":c5,"w":1},{"n":"RSI 40-70","ok":c6,"w":2},
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
            "dur":dur(trend,atr_v,p,t1l),"rsi":round(float(rv.iloc[-1]),1),
            "adx":round(float(adv.iloc[-1]),1),"vr":round(vr,1),"exp":vr>=3.0,
            "liq":liq,"liq_lbl":liq_lbl,"liq_unit":liq_unit,
            "rsv":rsv,"rsl":rsl,
            "obs":find_obs(dfd),"fvgs":find_fvgs(dfd),
            "conds":conds,"cok":sum(1 for c in conds if c["ok"]),"is_custom":False,
            "rank_score":0,"rank_reasons":[],"rank_pos":0,"rank_medal":"",
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
                return {"price":float(d["lastPrice"]),"chg":float(d["priceChangePercent"]),"vol_usdt":float(d.get("quoteVolume",0))}
    except: pass
    try:
        ys=sym.replace("USDT","-USD"); df=get_df(ys,"5d","1h")
        if not df.empty:
            p=float(df["Close"].iloc[-1]); p2=float(df["Close"].iloc[-24]) if len(df)>=24 else p
            return {"price":p,"chg":round((p-p2)/p2*100,2),"vol_usdt":0}
    except: pass
    return None

def analyze_crypto(sym,name,bdf=None):
    try:
        info=get_crypto_px(sym)
        if not info: return None
        price=info["price"]; chg=info["chg"]; vol_usdt=info.get("vol_usdt",0)
        ys=sym.replace("USDT","-USD"); dfd=get_df(ys,"2y","1d")
        if dfd.empty or len(dfd)<50: return None
        e20d=float(ema(dfd["Close"],20).iloc[-1])
        dfw=get_df(ys,"5y","1wk")
        e20w=float(ema(dfw["Close"],20).iloc[-1]) if len(dfw)>=20 else None
        ad=price>e20d; aw=price>e20w if e20w else False
        if ad and aw: trend,stars="سوينج",2
        elif ad: trend,stars="مضاربة",1
        else: trend,stars="تجنب",0
        d=dfd.copy(); rv=rsi_f(d["Close"])
        adv=adx_f(d["High"],d["Low"],d["Close"])
        av=atr_f(d["High"],d["Low"],d["Close"])
        atr_v=float(av.iloc[-1]); adx_v=round(float(adv.iloc[-1]),1); p=round(price,4)
        lb=round(p*0.995,4); t1s=round(p+atr_v*1.8,4); t1l=round(p+atr_v*2.0,4)
        t2s=round(p+atr_v*3.8,4); t2l=round(p+atr_v*4.0,4)
        ss=round(p-atr_v*1.3,4); sl=round(p-atr_v*1.5,4)
        rr=round((t1l-p)/max(p-sl,0.001),2)
        psl=round((p-sl)/p*100,2); ptp=round((t1l-p)/p*100,2)
        tp=round((atr_v*1.5/p)*100,2); tg=round(atr_v*0.3,4); be=round(p+(p-sl)*0.1,4)
        liq=round(vol_usdt/1e6,1)
        # ══ السيولة بالدولار للكريبتو ══
        liq_lbl=("عالية 🟢 "+str(liq)+" م.$") if liq>100 else (("متوسطة 🟡 "+str(liq)+" م.$") if liq>10 else ("منخفضة 🔴 "+str(liq)+" م.$"))
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
            "dur":dur(trend,atr_v,p,t1l),"rsi":round(float(rv.iloc[-1]),1),
            "adx":adx_v,"vr":round(abs(chg)/3,1) if abs(chg)>3 else 1,"exp":chg>10,
            "liq":liq,"liq_lbl":liq_lbl,"liq_unit":"م.$",
            "rsv":rsv,"rsl":rsl,
            "obs":find_obs(dfd),"fvgs":find_fvgs(dfd),
            "conds":[],"cok":0,"is_custom":False,
            "rank_score":0,"rank_reasons":[],"rank_pos":0,"rank_medal":"",
        }
    except: return None

TELEGRAM_CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),"telegram_config.json")

def load_telegram():
    if os.path.exists(TELEGRAM_CONFIG_FILE):
        with open(TELEGRAM_CONFIG_FILE,"r") as f:
            return json.load(f)
    return {"token":"","chat_id":"","enabled":False,"min_rank":70,"last_sent":{}}

def save_telegram(cfg):
    with open(TELEGRAM_CONFIG_FILE,"w") as f:
        json.dump(cfg,f,indent=2)

def send_telegram(message):
    cfg = load_telegram()
    if not cfg.get("enabled") or not cfg.get("token") or not cfg.get("chat_id"):
        return False
    try:
        import urllib.request, urllib.parse
        url = "https://api.telegram.org/bot"+cfg["token"]+"/sendMessage"
        data = urllib.parse.urlencode({
            "chat_id": cfg["chat_id"],
            "text": message,
            "parse_mode": "HTML"
        }).encode()
        req = urllib.request.Request(url, data)
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except: return False

def check_and_notify(results, market):
    cfg = load_telegram()
    if not cfg.get("enabled"): return
    min_rank = cfg.get("min_rank", 70)
    last_sent = cfg.get("last_sent", {})
    today = datetime.now().strftime("%Y-%m-%d")
    sent_count = 0
    mkt_names = {"tadawul":"🇸🇦 تاسي","us":"🇺🇸 أمريكي","crypto":"💰 عملات"}
    mkt_label = mkt_names.get(market, market)
    for s in results:
        if s.get("verdict") != "BUY": continue
        if s.get("rank_score",0) < min_rank: continue
        key = s["code"]+"-"+today
        if key in last_sent: continue
        medal = s.get("rank_medal","")
        stars = "⭐"*s.get("stars",0)
        msg = "⚡ جلال رادار — إشارة جديدة\n\n"
        msg += medal+" "+s["name"]+" ("+s["code"]+")\n"
        msg += "السوق: "+mkt_label+"\n"
        msg += "السعر: "+str(s["price"])+"\n"
        msg += "الترند: "+stars+" "+s.get("trend","")+"\n"
        msg += "JRF Score: "+str(s["score"])+"/20\n"
        msg += "الترجيح: "+str(s.get("rank_score",""))+"\n"
        msg += "TP1: "+str(s.get("t1l",""))+" (+"+str(s.get("ptp",""))+"%)\n"
        msg += "SL: "+str(s.get("sl",""))+" (-"+str(s.get("psl",""))+"%)\n"
        msg += "R:R: "+str(s.get("rr",""))+"\n"
        msg += "السبب: "+(" | ".join(s.get("rank_reasons",[])[:3]))
        if send_telegram(msg):
            last_sent[key] = datetime.now().strftime("%H:%M")
            sent_count += 1
    if sent_count > 0:
        cfg["last_sent"] = last_sent
        save_telegram(cfg)

def run_scan(market):
    scan_state[market]["status"]="scanning"
    scan_state[market]["progress"]=0
    custom=load_custom(); excl=custom.get("excluded",[])
    bdf=get_df(BENCHMARK.get(market,""),"2y","1d")
    if market=="tadawul": stocks={**DEFAULT_TADAWUL,**custom.get("tadawul",{})}
    elif market=="us": stocks={**DEFAULT_US,**custom.get("us",{})}
    else: stocks={**DEFAULT_CRYPTO,**custom.get("crypto",{})}
    stocks={k:v for k,v in stocks.items() if k not in excl}
    total=len(stocks); scan_state[market]["total"]=total
    results=[]; lock=threading.Lock(); done=[0]

    def scan_one(code,name):
        if market=="crypto": r=analyze_crypto(code,name,bdf)
        else: r=analyze(code,name,market,bdf)
        with lock:
            done[0]+=1
            scan_state[market]["progress"]=round(done[0]/total*100)
            if r:
                r["is_custom"]=code in custom.get(market,{})
                results.append(r)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs=[ex.submit(scan_one,c,n) for c,n in stocks.items()]
        for f in as_completed(futs): pass

    for s in results: calc_rank(s)
    results.sort(key=lambda x:(x["priority"],-x.get("rank_score",0)))
    medals=["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣","8️⃣","9️⃣","🔟"]
    for i,s in enumerate(results):
        s["rank_pos"]=i+1
        s["rank_medal"]=medals[i] if i<len(medals) else str(i+1)
    scan_state[market]["data"]=results[:20]
    scan_state[market]["last_scan"]=datetime.now().strftime("%Y-%m-%d %H:%M")
    scan_state[market]["status"]="done"
    scan_state[market]["progress"]=100
    try: check_and_notify(scan_state[market]["data"] or [], market)
    except: pass

def cv(v): return '<span class="cv" onclick="copyVal(this)">'+str(v)+'</span>'

def build_row(s,i):
    vl=s["verdict"].lower()
    sfx=".SR" if s["market"]=="tadawul" else ""
    bc="rb-buy" if s["verdict"]=="BUY" and "مشروط" not in s["bt"] else (
       "rb-cond" if "مشروط" in s["bt"] else (
       "rb-wait" if s["verdict"]=="WAIT" else "rb-avoid"))
    dc="pos" if s["above_daily"] else "neg"
    wc="pos" if s["above_weekly"] else "neg"
    mc="pos" if s["above_monthly"] else "neg"
    sc_cls="sc-hi" if s["score"]>=15 else ("sc-md" if s["score"]>=10 else "sc-lo")
    stars="⭐"*s["stars"]
    exp='<span class="exp-dot">🚀</span>' if s.get("exp") else ""
    rpos=s.get("rank_pos",99); medal=s.get("rank_medal","")
    rnk="rank-1" if rpos==1 else ("rank-2" if rpos==2 else ("rank-3" if rpos==3 else "rank-other"))
    chg_html=""
    if "chg" in s:
        chg=s["chg"]; cls="pos" if chg>=0 else "neg"
        chg_html='<span class="'+cls+'" style="font-size:0.65rem;">'+("+"+str(chg) if chg>=0 else str(chg))+'%</span>'

    row=(
        '<div class="srow '+vl+'" onclick="toggle(this.dataset.mkt,this.dataset.idx)" data-mkt="'+s["market"]+'" data-idx="'+str(i)+'">'
        '<div class="sr-l">'
        '<span class="'+bc+'">'+s["bt"]+'</span>'
        '<span class="sr-name">'+s["name"]+'</span>'
        '<span class="sr-code">'+s["code"]+sfx+'</span>'
        +exp+chg_html+
        '</div>'
        '<div class="sr-m"><span class="sr-stars">'+stars+'</span><span class="sr-tr">'+s["trend"]+'</span></div>'
        '<div class="sr-r">'
        '<span class="sr-price">'+str(s["price"])+'</span>'
        '<span class="sr-sc '+sc_cls+'">'+str(s["score"])+'/20</span>'
        '<span class="sr-fr">'
        '<span class="'+dc+'">'+("+" if s["above_daily"] else "—")+'</span>'
        '<span class="'+wc+'">'+("+" if s["above_weekly"] else "—")+'</span>'
        '<span class="'+mc+'">'+("+" if s["above_monthly"] else "—")+'</span>'
        '</span>'
        '<span class="sr-rsi">'+str(s["rsi"])+'</span>'
        '<span class="rank-badge '+rnk+'">'+medal+'</span>'
        '<span class="sr-chev" id="ch-'+s["market"]+'-'+str(i)+'">▾</span>'
        '</div>'
        '</div>'
    )

    why=""
    reasons=s.get("rank_reasons",[])
    if rpos and reasons:
        why='<div class="why-box">'
        why+='<div class="why-ttl">'+medal+' الترتيب #'+str(rpos)+' — نقطة الترجيح: '+str(s.get("rank_score",0))+'</div>'
        why+='<div class="why-tags">'
        for r in reasons:
            cls="why-tag warn" if "⚠️" in r else "why-tag"
            why+='<span class="'+cls+'">'+r+'</span>'
        why+='</div></div>'

    note_html='<div class="cond-note">⚠️ '+s["note"]+'</div>' if s.get("note") else ""

    ob_html=""
    if s.get("obs"):
        ob_html='<div class="ds"><div class="dt">📦 Order Blocks — مناطق دعم مؤسسية</div>'
        ob_html+='<div class="pro-exp">المنطقة اللي دخل منها المال الكبير — لو السعر رجع وارتد = دخول قوي 🎯</div>'
        for ob in s["obs"]:
            dist=round(((s["price"]-ob["mid"])/s["price"])*100,1)
            arrow="⬇️" if dist>0 else "⬆️"
            prox="قريب 🔥" if abs(dist)<3 else ("متوسط" if abs(dist)<8 else "بعيد")
            ob_html+='<div class="ob-row"><div><span class="ob-z">'+str(ob["low"])+" — "+str(ob["high"])+'</span>'
            ob_html+='<div class="ob-d">'+arrow+" "+str(abs(dist))+"% | "+prox+'</div></div>'
            ob_html+='<span class="ob-m">وسط: '+cv(ob["mid"])+'</span></div>'
        ob_html+='</div>'

    fvg_html=""
    if s.get("fvgs"):
        fvg_html='<div class="ds"><div class="dt">⚡ Fair Value Gaps — فجوات السعر</div>'
        fvg_html+='<div class="pro-exp">فجوة تركها السعر — غالباً يرجع يملأها قبل ما يكمل الصعود 📍</div>'
        for fvg in s["fvgs"]:
            dist=round(((s["price"]-fvg["mid"])/s["price"])*100,1)
            arrow="⬇️" if dist>0 else "⬆️"
            filled=s["price"]>=fvg["bottom"] and s["price"]<=fvg["top"]
            st="✅ داخلها" if filled else ("قريب 🔥" if abs(dist)<3 else "لم تُملأ")
            fvg_html+='<div class="ob-row"><div><span class="ob-z">'+str(fvg["bottom"])+" — "+str(fvg["top"])+'</span>'
            fvg_html+='<div class="ob-d">'+arrow+" "+str(abs(dist))+"% | "+st+'</div></div>'
            fvg_html+='<span class="ob-m">وسط: '+cv(fvg["mid"])+'</span></div>'
        fvg_html+='</div>'

    conds_html="".join(
        '<div class="ci '+("co" if c["ok"] else "cf")+'">'
        '<div class="cd '+("dok" if c["ok"] else "dfail")+'"></div>'
        '<span>'+c["n"]+'</span><span class="cw">★'+str(c["w"])+'</span></div>'
        for c in s.get("conds",[])
    )

    # ══ وحدة السيولة حسب السوق ══
    liq_unit_label = "م.ر = مليون ريال" if s["market"]=="tadawul" else "م.$ = مليون دولار"

    rs_cls="pos" if s.get("rsv",0)>2 else ("neg" if s.get("rsv",0)<-2 else "muted-t")
    del_btn='<button class="btn-del" onclick="event.stopPropagation();delStock(\''+s["code"]+'\',\''+s["market"]+'\')">🗑</button>' if s.get("is_custom") else ""
    excl_btn='<button class="btn-excl" onclick="event.stopPropagation();exclStock(\''+s["code"]+'\')">⊘ استبعاد</button>'

    detail=(
        '<div class="det" id="det-'+s["market"]+'-'+str(i)+'">'
        +why+note_html+
        '<div class="dq">'
        '<div class="dqi"><div class="dql">القوة النسبية 📊</div><div class="dqv '+rs_cls+'">'+s.get("rsl","—")+'</div><div class="dqh">مقارنة بمؤشر السوق</div></div>'
        '<div class="dqi"><div class="dql">السيولة اليومية 💧</div><div class="dqv">'+s.get("liq_lbl","—")+'</div><div class="dqh">'+liq_unit_label+'</div></div>'
        '<div class="dqi"><div class="dql">ADX — قوة الاتجاه</div><div class="dqv">'+str(s["adx"])+'</div><div class="dqh">&gt;40 قوي | 20-40 معقول | &lt;20 عرضي</div></div>'
        '<div class="dqi"><div class="dql">المدة المتوقعة ⏱</div><div class="dqv teal-t">'+s.get("dur","—")+'</div><div class="dqh">للوصول لـ TP1</div></div>'
        '</div>'
        '<div class="ds"><div class="dt">🛒 أمر الشراء</div>'
        '<div class="og1"><div class="oi"><div class="ol">سعر الأمر — Limit</div>'
        '<div class="ov ev">'+cv(s["lb"])+'</div><div class="oh">اضغط للنسخ</div></div></div></div>'
        '<div class="ds"><div class="dt">💰 جني الربح — Take Profit</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov tv">'+cv(s["t1s"])+'</div><div class="oh">Trigger Price</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov tv">'+cv(s["t1l"])+'</div><div class="oh pos">+'+str(s["ptp"])+'%</div></div>'
        '</div>'
        '<div class="og2" style="margin-top:5px;">'
        '<div class="oi"><div class="ol">إيقاف TP2</div><div class="ov tv">'+cv(s["t2s"])+'</div></div>'
        '<div class="oi"><div class="ol">أمر TP2</div><div class="ov tv">'+cv(s["t2l"])+'</div><div class="oh pos">+'+str(round(s["ptp"]*2,2))+'%</div></div>'
        '</div></div>'
        '<div class="ds ds-sl"><div class="dt">🛡 وقف الخسارة — Stop Loss</div>'
        '<div class="og2">'
        '<div class="oi"><div class="ol">سعر الإيقاف</div><div class="ov sv">'+cv(s["ss"])+'</div><div class="oh">Trigger Price</div></div>'
        '<div class="oi"><div class="ol">سعر الأمر</div><div class="ov sv">'+cv(s["sl"])+'</div><div class="oh neg">-'+str(s["psl"])+'%</div></div>'
        '</div>'
        '<div class="rr-row"><span>نسبة المخاطرة/العائد R:R: <strong style="color:var(--gold)">'+str(s["rr"])+'</strong> <span class="muted-t">(1.3+ جيد)</span></span><span>⏱ '+s.get("dur","—")+'</span></div>'
        '</div>'
        '<div class="ds ds-tr"><div class="dt">🚀 وقف خسارة متحرك — Trailing Stop</div>'
        '<div class="og3">'
        '<div class="oi"><div class="ol">نوع التتبع</div><div class="ov" style="color:var(--teal)">نسبة مئوية</div></div>'
        '<div class="oi"><div class="ol">المبلغ / النسبة</div><div class="ov sv">'+cv(str(s["tp"])+"%")+'</div><div class="oh">أدخل هذه النسبة</div></div>'
        '<div class="oi"><div class="ol">الفارق السعري</div><div class="ov muted-v">'+cv(s["tg"])+'</div></div>'
        '</div>'
        '<div class="tr-note">الدخول: '+str(s["price"])+'  |  الهدف الأقصى: '+str(s["t2l"])+'</div>'
        '</div>'
        '<div class="ds"><div class="dt">📊 إدارة المركز — Partial Exit</div>'
        '<div class="peg">'
        '<div class="pei"><div class="pel">عند TP1 — بيع</div><div class="pev pos">50%</div></div>'
        '<div class="pei"><div class="pel">عند TP2 — بيع</div><div class="pev pos">30%</div></div>'
        '<div class="pei"><div class="pel">اترك تجري</div><div class="pev teal-t">20%</div></div>'
        '</div>'
        '<div class="be-n">🔒 نقطة التعادل Break Even: حرك SL إلى '+str(s["be"])+'  بعد وصول TP1</div>'
        '</div>'
        +ob_html+fvg_html+
        ('<details onclick="event.stopPropagation()"><summary class="cb">📊 الشروط ('+str(s["cok"])+'/12)</summary>'
         '<div class="cleg">★★=2 نقطة  |  ★=1 نقطة</div><div class="cg">'+conds_html+'</div></details>'
         if conds_html else "")
        +'<div class="btn-row">'+del_btn+excl_btn+'</div>'
        '</div>'
    )
    return row+detail

# ══ CSS & JS (نفس الكود الأصلي بدون تغيير) ══
CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;500;600;700&display=swap');
:root{--bg:#070b14;--card:#0d1525;--card2:#111e33;--bdr:#1a2d4a;--gold:#d4a843;--gold2:#f0c060;--green:#22c55e;--red:#ef4444;--yel:#eab308;--blue:#3b82f6;--teal:#06b6d4;--purple:#8b5cf6;--txt:#e2e8f0;--mut:#64748b;--mut2:#94a3b8;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--txt);font-family:'IBM Plex Sans Arabic',sans-serif;min-height:100vh;direction:rtl;}
body::before{content:'';position:fixed;inset:0;background:radial-gradient(ellipse 800px 600px at 10% 20%,rgba(212,168,67,0.04),transparent),radial-gradient(ellipse 600px 400px at 90% 80%,rgba(6,182,212,0.04),transparent);pointer-events:none;z-index:0;}
.wrap{position:relative;z-index:1;max-width:1100px;margin:0 auto;padding:16px;}
.hdr{text-align:center;padding:28px 20px 18px;margin-bottom:16px;position:relative;}
.hdr::after{content:'';position:absolute;bottom:0;left:50%;transform:translateX(-50%);width:200px;height:1px;background:linear-gradient(90deg,transparent,var(--gold),transparent);}
.logo{font-size:clamp(1.6rem,5vw,2.8rem);font-weight:700;letter-spacing:3px;background:linear-gradient(135deg,var(--gold) 0%,var(--gold2) 40%,var(--teal) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:6px;}
.logo-sub{color:var(--mut);font-size:0.75rem;letter-spacing:1px;}
.sess-bar{display:inline-flex;align-items:center;gap:6px;margin-top:10px;background:var(--card);border:1px solid var(--bdr);border-radius:20px;padding:4px 14px;font-size:0.72rem;color:var(--mut2);}
.tabs{display:flex;gap:4px;background:var(--card);border:1px solid var(--bdr);border-radius:12px;padding:4px;margin-bottom:16px;}
.tab-btn{flex:1;padding:8px 12px;border-radius:8px;border:none;background:transparent;color:var(--mut);font-family:inherit;font-size:0.85rem;cursor:pointer;transition:all 0.2s;text-align:center;}
.tab-btn:hover{color:var(--txt);}
.tab-btn.active{background:linear-gradient(135deg,#1a3060,#0f1e40);color:var(--gold2);box-shadow:0 2px 8px rgba(0,0,0,0.3);}
.tab-c{display:none;}.tab-c.active{display:block;}
.scan-wrap{text-align:center;padding:8px 0 14px;}
.scan-btn{display:inline-flex;align-items:center;gap:8px;background:linear-gradient(135deg,#1a3060 0%,#0f1e40 100%);border:1px solid var(--gold);color:var(--gold2);padding:10px 28px;border-radius:50px;font-size:0.9rem;font-family:inherit;cursor:pointer;transition:all 0.3s;}
.scan-btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(212,168,67,0.2);}
.scan-ts{display:block;font-size:0.65rem;color:var(--mut);margin-top:4px;}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px;}
.chip{flex:1;min-width:72px;background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:8px 10px;text-align:center;cursor:pointer;transition:all 0.2s;}
.chip .num{font-size:1.4rem;font-weight:700;line-height:1;}.chip .lbl{font-size:0.61rem;color:var(--mut);margin-top:3px;}
.cg1{border-color:rgba(212,168,67,0.3)}.cg1 .num{color:var(--gold)}.cg1.af{background:rgba(212,168,67,0.07);}
.cg2{border-color:rgba(34,197,94,0.3)}.cg2 .num{color:var(--green)}.cg2.af{background:rgba(34,197,94,0.07);}
.cg3{border-color:rgba(234,179,8,0.3)}.cg3 .num{color:var(--yel)}.cg3.af{background:rgba(234,179,8,0.07);}
.cg4{border-color:rgba(239,68,68,0.3)}.cg4 .num{color:var(--red)}.cg4.af{background:rgba(239,68,68,0.07);}
.cg5{border-color:rgba(6,182,212,0.3)}.cg5 .num{color:var(--teal)}.cg5.af{background:rgba(6,182,212,0.07);}
.exp-sec{background:rgba(212,168,67,0.04);border:1px solid rgba(212,168,67,0.15);border-radius:11px;padding:11px;margin-bottom:13px;}
.exp-ttl{color:var(--gold);font-size:0.86rem;font-weight:600;margin-bottom:8px;}
.exp-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:6px;}
.exp-card{background:rgba(0,0,0,0.2);border:1px solid rgba(212,168,67,0.12);border-radius:8px;padding:7px 10px;display:flex;align-items:center;justify-content:space-between;gap:6px;}
.exp-nm{font-weight:600;font-size:0.83rem;}.exp-cd{color:var(--mut);font-size:0.63rem;}.exp-px{color:var(--gold2);font-weight:700;font-size:0.86rem;}.exp-rt{background:rgba(212,168,67,0.1);color:var(--yel);padding:2px 7px;border-radius:6px;font-size:0.67rem;font-weight:600;}
.heat-sec{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:12px;margin-bottom:14px;}
.heat-ttl{color:var(--gold);font-size:0.84rem;font-weight:600;margin-bottom:10px;}
.heat-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:7px;}
.heat-card{border-radius:9px;padding:10px;text-align:center;}
.heat-nm{font-size:0.76rem;font-weight:600;margin-bottom:5px;}.heat-sc{font-size:1.25rem;font-weight:700;margin-bottom:4px;}
.heat-bar{height:3px;border-radius:3px;margin:0 auto 4px;max-width:80%;}.heat-inf{font-size:0.6rem;color:rgba(255,255,255,0.5);}
.slist{display:flex;flex-direction:column;gap:4px;}
.srow{background:var(--card);border:1px solid var(--bdr);border-radius:10px;padding:10px 14px;cursor:pointer;transition:all 0.18s;display:flex;align-items:center;justify-content:space-between;gap:10px;}
.srow:hover{border-color:rgba(212,168,67,0.3);background:var(--card2);}
.srow.hidden{display:none;}.srow.buy{border-right:2px solid var(--green);}.srow.wait{border-right:2px solid var(--yel);}.srow.avoid{border-right:2px solid var(--red);}
.sr-l{display:flex;align-items:center;gap:6px;min-width:0;flex:1;}.sr-m{display:flex;align-items:center;gap:5px;flex-shrink:0;}.sr-r{display:flex;align-items:center;gap:7px;flex-shrink:0;}
.sr-name{font-weight:600;font-size:0.9rem;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:110px;}
.sr-code{color:var(--mut);font-size:0.62rem;flex-shrink:0;}.sr-stars{color:var(--gold);font-size:0.74rem;}.sr-tr{color:var(--mut);font-size:0.67rem;}
.sr-price{color:var(--gold2);font-weight:700;font-size:0.9rem;}
.sr-sc{font-weight:700;font-size:0.75rem;padding:2px 6px;border-radius:6px;}
.sc-hi{color:var(--green);background:rgba(34,197,94,0.1);}.sc-md{color:var(--yel);background:rgba(234,179,8,0.1);}.sc-lo{color:var(--red);background:rgba(239,68,68,0.1);}
.sr-fr{font-size:0.72rem;display:flex;gap:2px;font-weight:600;}.sr-rsi{color:var(--mut);font-size:0.66rem;}
.sr-chev{color:var(--mut);font-size:0.65rem;transition:transform 0.25s;}.sr-chev.op{transform:rotate(180deg);}
.exp-dot{font-size:0.7rem;}
.rb-buy{background:rgba(34,197,94,0.1);color:var(--green);border:1px solid rgba(34,197,94,0.3);padding:2px 7px;border-radius:8px;font-size:0.66rem;font-weight:600;white-space:nowrap;flex-shrink:0;}
.rb-cond{background:rgba(234,179,8,0.1);color:var(--yel);border:1px solid rgba(234,179,8,0.3);padding:2px 7px;border-radius:8px;font-size:0.66rem;font-weight:600;white-space:nowrap;flex-shrink:0;}
.rb-wait{background:rgba(234,179,8,0.07);color:var(--yel);border:1px solid rgba(234,179,8,0.2);padding:2px 7px;border-radius:8px;font-size:0.66rem;white-space:nowrap;flex-shrink:0;}
.rb-avoid{background:rgba(239,68,68,0.1);color:var(--red);border:1px solid rgba(239,68,68,0.3);padding:2px 7px;border-radius:8px;font-size:0.66rem;white-space:nowrap;flex-shrink:0;}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;width:24px;height:24px;border-radius:50%;font-size:0.78rem;font-weight:700;flex-shrink:0;}
.rank-1{background:linear-gradient(135deg,#fbbf24,#d97706);color:#000;box-shadow:0 2px 8px rgba(251,191,36,0.4);}
.rank-2{background:linear-gradient(135deg,#94a3b8,#64748b);color:#000;}
.rank-3{background:linear-gradient(135deg,#cd7f32,#92400e);color:#fff;}
.rank-other{background:rgba(100,116,139,0.15);color:var(--mut);border:1px solid var(--bdr);}
.det{display:none;background:rgba(0,0,0,0.15);border:1px solid var(--bdr);border-radius:0 0 10px 10px;padding:13px;margin-top:-4px;border-top:none;}
.det.op{display:block;}
.why-box{background:rgba(59,130,246,0.05);border:1px solid rgba(59,130,246,0.15);border-radius:9px;padding:10px;margin-bottom:10px;}
.why-ttl{font-size:0.66rem;color:var(--blue);font-weight:600;margin-bottom:7px;}
.why-tags{display:flex;flex-wrap:wrap;gap:5px;}
.why-tag{background:rgba(0,0,0,0.2);border:1px solid var(--bdr);border-radius:6px;padding:2px 8px;font-size:0.62rem;color:var(--txt);}
.why-tag.warn{border-color:rgba(234,179,8,0.3);color:var(--yel);}
.dq{display:grid;grid-template-columns:repeat(4,1fr);gap:7px;margin-bottom:11px;}
.dqi{background:rgba(0,0,0,0.2);border-radius:8px;padding:7px;text-align:center;border:1px solid rgba(255,255,255,0.04);}
.dql{font-size:0.58rem;color:var(--mut);margin-bottom:2px;font-weight:500;}.dqv{font-size:0.78rem;font-weight:600;}.dqh{font-size:0.52rem;color:var(--mut);margin-top:2px;}
.ds{background:rgba(0,0,0,0.18);border:1px solid var(--bdr);border-radius:9px;padding:9px;margin-bottom:7px;}
.ds-sl{border-color:rgba(239,68,68,0.18);background:rgba(239,68,68,0.02);}
.ds-tr{border-color:rgba(6,182,212,0.18);background:rgba(6,182,212,0.02);}
.dt{font-size:0.63rem;color:var(--mut2);font-weight:600;margin-bottom:7px;}
.og1{display:grid;grid-template-columns:1fr;gap:5px;}.og2{display:grid;grid-template-columns:1fr 1fr;gap:5px;}.og3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;}
.oi{text-align:center;}.ol{font-size:0.57rem;color:var(--mut);margin-bottom:2px;}
.ov{font-size:0.88rem;font-weight:700;padding:2px 4px;border-radius:4px;display:inline-block;}
.ev{color:var(--txt)}.tv{color:var(--green)}.sv{color:var(--red)}.muted-v{color:var(--mut)}
.oh{font-size:0.55rem;color:var(--mut);margin-top:1px;}
.cv{cursor:pointer;border-radius:4px;padding:1px 3px;transition:all 0.12s;}.cv:hover{background:rgba(212,168,67,0.12);}.cv:active{transform:scale(0.92);}.copied{background:rgba(34,197,94,0.15)!important;color:var(--green)!important;}
.rr-row{display:flex;justify-content:space-between;font-size:0.66rem;color:var(--mut);margin-top:7px;padding-top:6px;border-top:1px solid var(--bdr);}
.tr-note{font-size:0.6rem;color:var(--teal);margin-top:5px;padding-top:5px;border-top:1px solid rgba(6,182,212,0.15);}
.peg{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px;margin-bottom:7px;}
.pei{text-align:center;background:rgba(0,0,0,0.18);border-radius:6px;padding:5px;}.pel{font-size:0.57rem;color:var(--mut);}.pev{font-size:0.88rem;font-weight:700;margin-top:1px;}
.be-n{font-size:0.62rem;color:var(--teal);background:rgba(6,182,212,0.04);padding:5px 8px;border-radius:5px;border:1px solid rgba(6,182,212,0.1);}
.pro-exp{font-size:0.62rem;color:var(--mut);background:rgba(59,130,246,0.04);padding:4px 7px;border-radius:5px;margin-bottom:6px;border-right:2px solid var(--blue);}
.cond-note{font-size:0.63rem;color:var(--yel);background:rgba(234,179,8,0.05);padding:3px 7px;border-radius:5px;margin-bottom:7px;border-right:2px solid var(--yel);}
.ob-row{display:flex;justify-content:space-between;align-items:flex-start;padding:3px 5px;border-radius:5px;background:rgba(59,130,246,0.04);margin-bottom:3px;font-size:0.66rem;}
.ob-z{color:var(--blue)}.ob-m{color:var(--mut);font-size:0.6rem;}.ob-d{font-size:0.57rem;color:var(--mut);margin-top:1px;}
.cb{background:none;border:1px solid var(--bdr);color:var(--mut);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.63rem;cursor:pointer;width:100%;margin-top:3px;}
.cleg{font-size:0.57rem;color:var(--mut);margin-top:5px;opacity:0.6;}
.cg{display:grid;grid-template-columns:1fr 1fr;gap:2px;margin-top:5px;}
.ci{display:flex;align-items:center;gap:3px;font-size:0.6rem;padding:2px 4px;border-radius:4px;background:rgba(255,255,255,0.02);}
.co{color:var(--green)}.cf{color:var(--red);opacity:0.5;}
.cd{width:4px;height:4px;border-radius:50%;flex-shrink:0;}.dok{background:var(--green)}.dfail{background:var(--red);opacity:0.4;}.cw{color:var(--gold);font-size:0.55rem;margin-right:auto;}
.btn-row{display:flex;gap:5px;margin-top:7px;}
.btn-del{background:rgba(239,68,68,0.07);border:1px solid rgba(239,68,68,0.2);color:var(--red);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.63rem;cursor:pointer;}
.btn-excl{background:rgba(100,116,139,0.07);border:1px solid rgba(100,116,139,0.2);color:var(--mut);padding:4px 9px;border-radius:5px;font-family:inherit;font-size:0.63rem;cursor:pointer;}
.pos{color:var(--green)}.neg{color:var(--red)}.muted-t{color:var(--mut)}.teal-t{color:var(--teal);}
.cp{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px;margin-bottom:13px;}
.cp-ttl{font-size:0.8rem;color:var(--teal);margin-bottom:8px;font-weight:600;}
.ci-row{display:flex;gap:5px;flex-wrap:wrap;margin-bottom:7px;}
.ci-inp{background:rgba(0,0,0,0.3);border:1px solid var(--bdr);color:var(--txt);padding:6px 9px;border-radius:7px;font-family:inherit;font-size:0.77rem;flex:1;min-width:80px;}
.ci-inp:focus{outline:none;border-color:var(--gold);}
.add-btn{background:rgba(34,197,94,0.09);border:1px solid rgba(34,197,94,0.3);color:var(--green);padding:6px 12px;border-radius:7px;cursor:pointer;font-family:inherit;font-size:0.77rem;}
.file-row{display:flex;align-items:center;gap:7px;margin-top:5px;}
.file-lbl{background:rgba(59,130,246,0.09);border:1px solid rgba(59,130,246,0.25);color:var(--blue);padding:5px 11px;border-radius:7px;cursor:pointer;font-size:0.73rem;white-space:nowrap;}
.file-hint{font-size:0.64rem;color:var(--mut);}
.ecl{display:inline-block;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.18);color:var(--red);padding:2px 7px;border-radius:10px;font-size:0.63rem;cursor:pointer;margin:2px;}
.gloss{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:11px;margin-bottom:13px;}
.gloss-ttl{font-size:0.8rem;font-weight:600;color:var(--teal);cursor:pointer;display:flex;justify-content:space-between;align-items:center;}
.gloss-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:5px;margin-top:10px;}
.gi{background:rgba(0,0,0,0.2);border-radius:7px;padding:7px 9px;border-right:2px solid var(--blue);}
.gi-en{font-size:0.73rem;font-weight:700;color:var(--blue);}.gi-ar{font-size:0.7rem;color:var(--txt);margin-top:1px;}.gi-desc{font-size:0.62rem;color:var(--mut);margin-top:2px;line-height:1.4;}
.lo{position:fixed;inset:0;background:rgba(7,11,20,0.95);display:none;flex-direction:column;align-items:center;justify-content:center;z-index:100;}
.lo.show{display:flex;}
.lo-spin{width:52px;height:52px;border:2px solid var(--bdr);border-top-color:var(--gold);border-radius:50%;animation:spin 1s linear infinite;margin-bottom:14px;}
@keyframes spin{to{transform:rotate(360deg)}}
.lo-txt{color:var(--gold2);font-size:0.88rem;}.lo-sub{color:var(--mut);font-size:0.68rem;margin-top:4px;}
.lo-prog{width:200px;height:3px;background:var(--bdr);border-radius:3px;margin-top:12px;overflow:hidden;}
.lo-pfill{height:100%;background:linear-gradient(90deg,var(--gold),var(--teal));border-radius:3px;transition:width 0.5s;}
.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:linear-gradient(135deg,var(--gold),var(--gold2));color:#000;padding:6px 16px;border-radius:14px;font-size:0.74rem;font-weight:600;opacity:0;transition:opacity 0.3s;z-index:200;pointer-events:none;}
.toast.show{opacity:1;}
.bt-sec{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:13px;margin-bottom:14px;}
.bt-ttl{color:var(--gold);font-size:0.84rem;font-weight:600;margin-bottom:10px;display:flex;justify-content:space-between;align-items:center;}
.bt-btn{background:rgba(212,168,67,0.1);border:1px solid rgba(212,168,67,0.3);color:var(--gold2);padding:5px 14px;border-radius:7px;font-family:inherit;font-size:0.74rem;cursor:pointer;}
.bt-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:7px;}
.bt-card{background:rgba(0,0,0,0.2);border-radius:9px;padding:10px;border:1px solid var(--bdr);}
.bt-name{font-size:0.8rem;font-weight:600;margin-bottom:6px;}
.bt-stats{display:grid;grid-template-columns:1fr 1fr;gap:4px;}
.bt-stat{text-align:center;}.bt-sv{font-size:1.1rem;font-weight:700;}.bt-sl{font-size:0.58rem;color:var(--mut);}
.bt-grade{display:inline-block;padding:2px 8px;border-radius:6px;font-size:0.72rem;font-weight:700;}
.bg-A{background:rgba(34,197,94,0.15);color:var(--green);}.bg-B{background:rgba(212,168,67,0.15);color:var(--gold2);}.bg-C{background:rgba(234,179,8,0.15);color:var(--yel);}.bg-D{background:rgba(239,68,68,0.15);color:var(--red);}
.tg-sec{background:var(--card);border:1px solid var(--bdr);border-radius:11px;padding:13px;margin-bottom:14px;}
.tg-ttl{color:var(--teal);font-size:0.84rem;font-weight:600;margin-bottom:10px;}
.tg-row{display:flex;gap:7px;align-items:center;margin-bottom:8px;flex-wrap:wrap;}
.tg-lbl{font-size:0.72rem;color:var(--mut);min-width:100px;}
.tg-inp{background:rgba(0,0,0,0.3);border:1px solid var(--bdr);color:var(--txt);padding:6px 9px;border-radius:7px;font-family:inherit;font-size:0.77rem;flex:1;}
.tg-inp:focus{outline:none;border-color:var(--teal);}
.tg-save{background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);color:var(--teal);padding:6px 14px;border-radius:7px;font-family:inherit;font-size:0.77rem;cursor:pointer;}
.tg-test{background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);color:var(--green);padding:6px 14px;border-radius:7px;font-family:inherit;font-size:0.77rem;cursor:pointer;}
.nr{text-align:center;padding:50px 20px;color:var(--mut);}.nr-icon{font-size:2.5rem;margin-bottom:10px;}
@media(max-width:768px){.wrap{padding:10px;}.sr-m{display:none;}.sr-rsi{display:none;}.dq{grid-template-columns:1fr 1fr;}.og2,.og3{grid-template-columns:1fr 1fr;}.peg{grid-template-columns:1fr 1fr;}.cg{grid-template-columns:1fr;}.exp-grid{grid-template-columns:1fr;}.heat-grid{grid-template-columns:repeat(2,1fr);}.tabs{gap:2px;}.tab-btn{padding:7px 8px;font-size:0.78rem;}.gloss-grid{grid-template-columns:1fr;}.sr-name{max-width:85px;}}
@media(max-width:400px){.sr-sc{display:none;}.og2{grid-template-columns:1fr;}.tabs{flex-wrap:wrap;}}
</style>"""

JS = r"""<script>
var aTab='tadawul',aFilt={tadawul:null,us:null,crypto:null};
var scanPoll=null;
function switchTab(t){aTab=t;document.querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));document.querySelectorAll('.tab-c').forEach(c=>c.classList.remove('active'));document.getElementById('tb-'+t).classList.add('active');document.getElementById('tc-'+t).classList.add('active');}
function toggle(mkt,i){var det=document.getElementById('det-'+mkt+'-'+i);var ch=document.getElementById('ch-'+mkt+'-'+i);if(!det)return;if(det.classList.contains('op')){det.classList.remove('op');if(ch)ch.classList.remove('op');}else{det.classList.add('op');if(ch)ch.classList.add('op');}}
function filterChip(mkt,ft,el){var chips=document.querySelectorAll('#chips-'+mkt+' .chip');var rows=document.querySelectorAll('#list-'+mkt+' .srow');if(aFilt[mkt]===ft){aFilt[mkt]=null;chips.forEach(c=>c.classList.remove('af'));rows.forEach(r=>r.classList.remove('hidden'));}else{aFilt[mkt]=ft;chips.forEach(c=>c.classList.remove('af'));el.classList.add('af');rows.forEach(r=>{var show=false;if(ft==='all')show=true;if(ft==='buy'&&r.classList.contains('buy'))show=true;if(ft==='wait'&&r.classList.contains('wait'))show=true;if(ft==='avoid'&&r.classList.contains('avoid'))show=true;if(ft==='exp'&&r.querySelector('.exp-dot'))show=true;r.classList.toggle('hidden',!show);});}}
function startScan(){var lb={tadawul:'تاسي',us:'السوق الأمريكي',crypto:'العملات'};document.getElementById('lo').classList.add('show');document.getElementById('lm').textContent=lb[aTab]||aTab;document.getElementById('lo-pfill').style.width='0%';fetch('/scan?market='+aTab).then(r=>r.json()).then(()=>{scanPoll=setInterval(pollProgress,1500);});}
function pollProgress(){fetch('/status?market='+aTab).then(r=>r.json()).then(d=>{var pct=d.progress||0;document.getElementById('lo-pfill').style.width=pct+'%';document.getElementById('lo-sub').textContent='تم مسح '+d.progress+'% — '+d.total+' سهم';if(d.status==='done'){clearInterval(scanPoll);sessionStorage.setItem('activeTab',aTab);window.location.reload();}});}
(function(){var saved=sessionStorage.getItem('activeTab');if(saved){sessionStorage.removeItem('activeTab');document.addEventListener('DOMContentLoaded',function(){if(saved!=='tadawul')switchTab(saved);});}})();
function copyVal(el){var t=el.textContent.trim();navigator.clipboard.writeText(t).then(()=>{el.classList.add('copied');showToast('✓ '+t);setTimeout(()=>el.classList.remove('copied'),1300);});}
function showToast(m){var t=document.getElementById('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),2000);}
function toggleGloss(){var b=document.getElementById('gloss-body');var c=document.getElementById('gloss-chev');if(b.style.display==='none'){b.style.display='block';c.textContent='▲';}else{b.style.display='none';c.textContent='▼';}}
function addStock(){var code=document.getElementById('nCode').value.trim().toUpperCase();var name=document.getElementById('nName').value.trim();var mkt=document.getElementById('nMkt').value;if(!code||!name){showToast('أدخل الرمز والاسم');return;}fetch('/add_stock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code,name,market:mkt})}).then(r=>r.json()).then(d=>{if(d.ok){showToast('تمت الإضافة ✓');setTimeout(()=>location.reload(),800);}});}
function uploadFile(){var file=document.getElementById('fileInput').files[0];var mkt=document.getElementById('nMkt').value;if(!file)return;var reader=new FileReader();reader.onload=function(e){var stocks=[];e.target.result.split(String.fromCharCode(10)).forEach(line=>{line=line.trim();if(!line)return;var p=line.split(/\s+/);if(p.length>=2)stocks.push({code:p[0].toUpperCase(),name:p.slice(1).join(' ')});});if(!stocks.length){showToast('ما في بيانات');return;}fetch('/add_bulk',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({stocks,market:mkt})}).then(r=>r.json()).then(d=>{showToast('تمت إضافة '+d.count+' ✓');setTimeout(()=>location.reload(),1000);});};reader.readAsText(file);}
function delStock(code,mkt){if(!confirm('حذف '+code+'؟'))return;fetch('/delete_stock',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code,market:mkt})}).then(r=>r.json()).then(d=>{if(d.ok)location.reload();});}
function exclStock(code){fetch('/exclude',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})}).then(r=>r.json()).then(d=>{if(d.ok)location.reload();});}
function runBacktest(mkt){showToast('جاري اختبار الاستراتيجية...');fetch('/backtest?market='+mkt).then(r=>r.json()).then(()=>{setTimeout(()=>loadBacktest(mkt),3000);});}
function loadBacktest(mkt){fetch('/backtest_results?market='+mkt).then(r=>r.json()).then(d=>{if(!d.data||!d.data.length){showToast('جاري الحساب...');setTimeout(()=>loadBacktest(mkt),5000);return;}var html='';d.data.forEach(function(s){var gc='bg-'+s.grade;var wr_cls=s.win_rate>=65?'pos':(s.win_rate>=50?'teal-t':'neg');html+='<div class="bt-card"><div class="bt-name" style="display:flex;justify-content:space-between;">'+s.name+'<span class="bt-grade '+gc+'">'+s.grade+'</span></div><div class="bt-stats"><div class="bt-stat"><div class="bt-sv '+wr_cls+'">'+s.win_rate+'%</div><div class="bt-sl">نسبة النجاح</div></div><div class="bt-stat"><div class="bt-sv">'+s.trades+'</div><div class="bt-sl">صفقة مختبرة</div></div><div class="bt-stat"><div class="bt-sv pos">+'+s.avg_win+'%</div><div class="bt-sl">متوسط الربح</div></div><div class="bt-stat"><div class="bt-sv neg">'+s.avg_loss+'%</div><div class="bt-sl">متوسط الخسارة</div></div></div></div>';});document.getElementById('bt-results-'+mkt).innerHTML=html;document.getElementById('bt-ts-'+mkt).textContent='آخر اختبار: '+d.ts;});}
function loadTelegramConfig(){fetch('/telegram_config').then(r=>r.json()).then(d=>{document.getElementById('tg-token').value=d.token||'';document.getElementById('tg-chat').value=d.chat_id||'';document.getElementById('tg-rank').value=d.min_rank||70;document.getElementById('tg-enabled').checked=d.enabled||false;});}
function saveTelegramConfig(){var data={token:document.getElementById('tg-token').value,chat_id:document.getElementById('tg-chat').value,min_rank:document.getElementById('tg-rank').value,enabled:document.getElementById('tg-enabled').checked};fetch('/telegram_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json()).then(d=>{if(d.ok)showToast('تم الحفظ ✓');});}
function testTelegram(){fetch('/test_telegram').then(r=>r.json()).then(d=>{showToast(d.ok?'✅ وصلت الرسالة!':'❌ تحقق من الإعدادات');});}
document.addEventListener('DOMContentLoaded',function(){loadTelegramConfig();});
function inclStock(code){fetch('/include',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})}).then(r=>r.json()).then(d=>{if(d.ok)location.reload();});}
</script>"""

GLOSSARY = [
    ("RSI","مؤشر القوة النسبية","يقيس زخم السعر. 40-70 مثالي للدخول. أقل من 30 ذروة بيع. أكثر من 70 ذروة شراء"),
    ("MACD","تقاطع المتوسطات","يكشف تغيير الزخم. MACD > Signal = إشارة صاعدة ✅"),
    ("EMA","المتوسط المتحرك الأسي","يتبع السعر بسرعة. فوقه = صاعد، تحته = هابط"),
    ("ADX","قوة الاتجاه","أقل 20 = سوق عرضي تجنب. 20-40 = ترند معقول. 40+ = ترند قوي 🔥"),
    ("ATR","متوسط المدى الحقيقي","يقيس تقلب السعر اليومي — أساس حساب SL وTP"),
    ("R:R","نسبة المخاطرة للعائد","1.3 = لكل ريال تخاطر تربح 1.3. كلما ارتفعت كانت الصفقة أفضل"),
    ("Order Block","منطقة دعم مؤسسية","المنطقة اللي دخل منها المال الكبير. رجوع السعر إليها = فرصة دخول قوية 🎯"),
    ("FVG","فجوة السعر","فجوة تركها السعر بسرعة — غالباً يرجع يملأها قبل ما يكمل الصعود 📍"),
    ("Trailing Stop","وقف خسارة متحرك","يتحرك مع السعر للأعلى — يحمي الأرباح ويترك الصفقة تكمل"),
    ("Break Even","نقطة التعادل","حرك SL لسعر الدخول بعد TP1 — تأمن الصفقة بدون خسارة"),
    ("م.$","مليون دولار","وحدة السيولة للسوق الأمريكي والعملات الرقمية"),
    ("م.ر","مليون ريال","وحدة السيولة للسوق السعودي"),
]

@app.route("/")
def index():
    custom=load_custom(); excl=custom.get("excluded",[])
    so,sl=is_us_session()
    out=["<!DOCTYPE html><html lang='ar' dir='rtl'><head>"]
    out.append('<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">')
    out.append('<title>جلال رادار v4.1</title>')
    out.append(CSS)
    out.append("</head><body>")
    out.append('<div id="toast" class="toast"></div>')
    out.append('<div class="lo" id="lo"><div class="lo-spin"></div>')
    out.append('<div class="lo-txt">🔍 جاري مسح <span id="lm"></span></div>')
    out.append('<div class="lo-sub" id="lo-sub">جاري التحضير...</div>')
    out.append('<div class="lo-prog"><div class="lo-pfill" id="lo-pfill" style="width:0%"></div></div></div>')
    out.append('<div class="wrap">')
    out.append('<div class="hdr"><div class="logo">⚡ جلال رادار</div>')
    out.append('<div class="logo-sub">Jalal Radar v4.1 — Halal Professional Edition</div>')
    out.append('<div class="sess-bar">🕐 '+sl+'</div></div>')
    out.append('<div class="tabs">')
    out.append('<button class="tab-btn active" id="tb-tadawul" onclick="switchTab(\'tadawul\')">🇸🇦 تاسي</button>')
    out.append('<button class="tab-btn" id="tb-us" onclick="switchTab(\'us\')">🇺🇸 أمريكي ('+str(len(DEFAULT_US))+')</button>')
    out.append('<button class="tab-btn" id="tb-crypto" onclick="switchTab(\'crypto\')">💰 عملات</button>')
    out.append('</div>')
    out.append('<div class="gloss"><div class="gloss-ttl" onclick="toggleGloss()">📖 دليل المصطلحات <span id="gloss-chev">▼</span></div>')
    out.append('<div id="gloss-body" style="display:none"><div class="gloss-grid">')
    for en,ar,desc in GLOSSARY:
        out.append('<div class="gi"><div class="gi-en">'+en+'</div><div class="gi-ar">'+ar+'</div><div class="gi-desc">'+desc+'</div></div>')
    out.append('</div></div></div>')
    out.append('<div class="cp"><div class="cp-ttl">➕ إضافة أسهم</div>')
    out.append('<div class="ci-row"><input class="ci-inp" id="nCode" placeholder="الرمز" style="max-width:105px;"><input class="ci-inp" id="nName" placeholder="الاسم">')
    out.append('<select class="ci-inp" id="nMkt" style="max-width:115px;"><option value="tadawul">🇸🇦 تاسي</option><option value="us">🇺🇸 أمريكي</option><option value="crypto">💰 عملات</option></select>')
    out.append('<button class="add-btn" onclick="addStock()">➕</button></div>')
    out.append('<div class="file-row"><label class="file-lbl" for="fileInput">📂 رفع ملف نصي</label>')
    out.append('<input type="file" id="fileInput" accept=".txt" style="display:none" onchange="uploadFile()">')
    out.append('<span class="file-hint">كل سطر: رمز اسم</span></div>')
    if excl:
        out.append('<div style="margin-top:7px;font-size:0.65rem;color:var(--mut);">مستبعدون:</div><div>')
        for code in excl:
            out.append('<span class="ecl" onclick="inclStock(\''+code+'\')">'+code+' ↩</span>')
        out.append('</div>')
    out.append('</div>')
    out.append('<div class="tg-sec"><div class="tg-ttl">📲 إعدادات تنبيهات Telegram</div>')
    out.append('<div class="tg-row"><span class="tg-lbl">Token البوت:</span><input class="tg-inp" id="tg-token" type="password" placeholder="123456:ABC..."></div>')
    out.append('<div class="tg-row"><span class="tg-lbl">Chat ID:</span><input class="tg-inp" id="tg-chat" placeholder="165932508"></div>')
    out.append('<div class="tg-row"><span class="tg-lbl">حد الترجيح:</span><input class="tg-inp" id="tg-rank" type="number" value="70" style="max-width:80px;"></div>')
    out.append('<div class="tg-row"><label style="display:flex;align-items:center;gap:7px;cursor:pointer;font-size:0.74rem;color:var(--mut2);"><input type="checkbox" id="tg-enabled"> تفعيل التنبيهات التلقائية</label></div>')
    out.append('<div class="tg-row"><button class="tg-save" onclick="saveTelegramConfig()">💾 حفظ</button><button class="tg-test" onclick="testTelegram()">🧪 اختبار</button></div></div>')

    for mkt,label,flag in [("tadawul","تاسي","🇸🇦"),("us","السوق الأمريكي","🇺🇸"),("crypto","العملات الرقمية","💰")]:
        data=scan_state[mkt]["data"] or []
        last=scan_state[mkt]["last_scan"] or ""
        acls=" active" if mkt=="tadawul" else ""
        out.append('<div class="tab-c'+acls+'" id="tc-'+mkt+'">')
        out.append('<div class="scan-wrap"><button class="scan-btn" onclick="startScan()">🔍 مسح '+flag+' '+label+'</button>')
        if last:
            total=scan_state[mkt].get("total",0)
            out.append('<span class="scan-ts">آخر مسح: '+last+' — تم مسح '+str(total)+' سهم، أفضل 20 نتيجة</span>')
        out.append('</div>')
        if data:
            buy_l=[s for s in data if s["verdict"]=="BUY"]
            wait_l=[s for s in data if s["verdict"]=="WAIT"]
            avoid_l=[s for s in data if s["verdict"]=="AVOID"]
            exp_l=[s for s in data if s.get("exp")]
            out.append('<div class="chips" id="chips-'+mkt+'">')
            out.append('<div class="chip cg1" onclick="filterChip(\''+mkt+'\',\'all\',this)"><div class="num">'+str(len(data))+'</div><div class="lbl">الكل</div></div>')
            out.append('<div class="chip cg2" onclick="filterChip(\''+mkt+'\',\'buy\',this)"><div class="num">'+str(len(buy_l))+'</div><div class="lbl">🟢 BUY</div></div>')
            out.append('<div class="chip cg3" onclick="filterChip(\''+mkt+'\',\'wait\',this)"><div class="num">'+str(len(wait_l))+'</div><div class="lbl">🟡 WAIT</div></div>')
            out.append('<div class="chip cg4" onclick="filterChip(\''+mkt+'\',\'avoid\',this)"><div class="num">'+str(len(avoid_l))+'</div><div class="lbl">🔴 AVOID</div></div>')
            out.append('<div class="chip cg5" onclick="filterChip(\''+mkt+'\',\'exp\',this)"><div class="num">'+str(len(exp_l))+'</div><div class="lbl">🚀 انفجار</div></div>')
            out.append('</div>')
            if exp_l:
                out.append('<div class="exp-sec"><div class="exp-ttl">🚀 تنبيهات الانفجار — حجم تداول شاذ</div><div class="exp-grid">')
                for s in exp_l:
                    sfx=".SR" if s["market"]=="tadawul" else ""
                    out.append('<div class="exp-card"><div><div class="exp-nm">'+s["name"]+'</div><div class="exp-cd">'+s["code"]+sfx+'</div></div>')
                    out.append('<div class="exp-px">'+str(s["price"])+'</div><div class="exp-rt">🚀 ×'+str(s["vr"])+'</div>')
                    out.append('<span style="font-size:0.67rem">'+s["bt"]+'</span></div>')
                out.append('</div></div>')
            heat=get_sector_heat(data,mkt)
            if heat:
                all_avgs=sorted([v["avg"] for v in heat.values()],reverse=True)
                top_t=max(1,len(all_avgs)//3)
                out.append('<div class="heat-sec"><div class="heat-ttl">🌡️ خريطة القطاعات</div><div class="heat-grid">')
                for sector,info in heat.items():
                    avg=info["avg"]; buy=info["buy"]; cnt=info["count"]
                    rank=list(all_avgs).index(avg) if avg in all_avgs else len(all_avgs)
                    if rank<top_t: bg="rgba(34,197,94,0.1)";bc="rgba(34,197,94,0.3)";sc2="var(--green)";lbl="🔥 الأقوى"
                    elif rank<top_t*2: bg="rgba(234,179,8,0.08)";bc="rgba(234,179,8,0.25)";sc2="var(--yel)";lbl="👀 متوسط"
                    else: bg="rgba(239,68,68,0.07)";bc="rgba(239,68,68,0.2)";sc2="var(--red)";lbl="❄️ الأضعف"
                    pct=min(int(avg/20*100),100)
                    out.append('<div class="heat-card" style="background:'+bg+';border:1px solid '+bc+';">')
                    out.append('<div class="heat-nm">'+sector+'</div><div class="heat-sc" style="color:'+sc2+'">'+str(avg)+'</div>')
                    out.append('<div class="heat-bar" style="background:'+sc2+';width:'+str(pct)+'%;"></div>')
                    out.append('<div class="heat-inf">'+lbl+' | '+str(buy)+' BUY من '+str(cnt)+'</div></div>')
                out.append('</div></div>')
            out.append('<div class="slist" id="list-'+mkt+'">')
            for i,s in enumerate(data):
                out.append(build_row(s,i))
            out.append('</div>')
        else:
            out.append('<div class="nr"><div class="nr-icon">📡</div><div>اضغط "مسح" لتحليل '+label+'</div></div>')
        out.append('<div class="bt-sec"><div class="bt-ttl"><span>📊 Backtest</span><button class="bt-btn" onclick="runBacktest(\''+mkt+'\');">🔬 ابدأ الاختبار</button></div>')
        out.append('<span class="scan-ts" id="bt-ts-'+mkt+'"></span><div class="bt-grid" id="bt-results-'+mkt+'"></div></div>')
        out.append('</div>')

    out.append('</div>'+JS+'</body></html>')
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
    return jsonify({"status":scan_state[m]["status"],"progress":scan_state[m].get("progress",0),"total":scan_state[m].get("total",0)})

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

@app.route("/backtest")
def backtest_route():
    market=request.args.get("market","tadawul")
    t=threading.Thread(target=run_backtest,args=(market,)); t.daemon=True; t.start()
    return jsonify({"status":"started"})

@app.route("/backtest_results")
def backtest_results():
    market=request.args.get("market","tadawul")
    cached=backtest_cache.get(market)
    if not cached: return jsonify({"data":[],"ts":""})
    return jsonify(cached)

@app.route("/telegram_config",methods=["GET","POST"])
def telegram_config():
    if request.method=="POST":
        d=request.get_json(); cfg=load_telegram()
        cfg["token"]=d.get("token",cfg.get("token",""))
        cfg["chat_id"]=d.get("chat_id",cfg.get("chat_id",""))
        cfg["enabled"]=d.get("enabled",False)
        cfg["min_rank"]=int(d.get("min_rank",70))
        save_telegram(cfg); return jsonify({"ok":True})
    cfg=load_telegram()
    return jsonify({"token":cfg.get("token",""),"chat_id":cfg.get("chat_id",""),
                    "enabled":cfg.get("enabled",False),"min_rank":cfg.get("min_rank",70)})

@app.route("/test_telegram")
def test_telegram():
    ok=send_telegram("✅ جلال رادار v4.1 — الاتصال يعمل بنجاح!")
    return jsonify({"ok":ok})

if __name__=="__main__":
    port=int(os.environ.get("PORT",5000))
    print("="*55)
    print("  ⚡ جلال رادار v4.1 — Halal Professional")
    print("  تاسي: "+str(len(DEFAULT_TADAWUL))+" سهم")
    print("  أمريكي: "+str(len(DEFAULT_US))+" سهم شرعي")
    print("  عملات: "+str(len(DEFAULT_CRYPTO))+" عملة")
    print("  http://localhost:5000")
    print("="*55)
    if port==5000: threading.Timer(1.5,lambda:webbrowser.open("http://localhost:5000")).start()
    app.run(host="0.0.0.0",port=port,debug=False)
