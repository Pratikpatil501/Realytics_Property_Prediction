"""
Mumbai House Price Prediction — Enhanced Flask API v3
Adds: POST /auth/register  POST /auth/login  GET /auth/profile
      POST /auth/logout     GET /admin/users  GET /admin/searches
Run: python app.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, json, os, uuid, hashlib, hmac, sqlite3, secrets
from datetime import datetime, timedelta

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))
CORS(app, supports_credentials=True, origins=["*"])

DB_PATH = "users.db"

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            phone TEXT, password TEXT NOT NULL, created_at TEXT NOT NULL, last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, guest_id TEXT,
            type TEXT NOT NULL, inputs TEXT NOT NULL, result_price REAL,
            region TEXT, created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            created_at TEXT NOT NULL, expires_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        """)

init_db()

model    = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
with open("meta.json") as f:
    meta = json.load(f)

# ─── Auth helpers ─────────────────────────────────────────────────────────────
def hash_password(pwd):
    salt = secrets.token_hex(16)
    key  = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000)
    return f"{salt}${key.hex()}"

def verify_password(pwd, stored):
    try:
        salt, key_hex = stored.split("$")
        key = hashlib.pbkdf2_hmac("sha256", pwd.encode(), salt.encode(), 100_000)
        return hmac.compare_digest(key.hex(), key_hex)
    except: return False

def create_session(user_id):
    token = secrets.token_urlsafe(48)
    expires = (datetime.utcnow() + timedelta(days=30)).isoformat()
    with get_db() as conn:
        conn.execute("INSERT INTO sessions VALUES (?,?,?,?)", (token, user_id, datetime.utcnow().isoformat(), expires))
    return token

def current_user():
    token = request.headers.get("X-Auth-Token")
    if not token: return None
    with get_db() as conn:
        row = conn.execute(
            "SELECT s.user_id, s.expires_at, u.* FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token=?", (token,)
        ).fetchone()
    if not row: return None
    if datetime.fromisoformat(row["expires_at"]) < datetime.utcnow(): return None
    return dict(row)

def log_search(stype, inputs, price=None, region=None):
    user = current_user()
    guest_id = request.headers.get("X-Guest-Id")
    with get_db() as conn:
        conn.execute("INSERT INTO searches (user_id,guest_id,type,inputs,result_price,region,created_at) VALUES (?,?,?,?,?,?,?)",
            (user["id"] if user else None, guest_id, stype, json.dumps(inputs), price, region, datetime.utcnow().isoformat()))

# ─── Region Data ──────────────────────────────────────────────────────────────
REGION_DATA = {
    "Colaba":{"safety":9,"connectivity":8,"pollution":5,"infra":9,"growth":8,"rental_yield":4.5,"zone":"South Mumbai"},
    "Cuffe Parade":{"safety":9,"connectivity":8,"pollution":5,"infra":9,"growth":8,"rental_yield":4.5,"zone":"South Mumbai"},
    "Malabar Hill":{"safety":10,"connectivity":7,"pollution":4,"infra":9,"growth":7,"rental_yield":3.8,"zone":"South Mumbai"},
    "Marine Lines":{"safety":8,"connectivity":9,"pollution":6,"infra":8,"growth":7,"rental_yield":4.2,"zone":"South Mumbai"},
    "Churchgate":{"safety":8,"connectivity":10,"pollution":7,"infra":8,"growth":6,"rental_yield":4.0,"zone":"South Mumbai"},
    "Fort":{"safety":7,"connectivity":10,"pollution":8,"infra":7,"growth":6,"rental_yield":4.5,"zone":"South Mumbai"},
    "Peddar Road":{"safety":9,"connectivity":7,"pollution":4,"infra":9,"growth":7,"rental_yield":3.5,"zone":"South Mumbai"},
    "Tardeo":{"safety":8,"connectivity":8,"pollution":6,"infra":8,"growth":7,"rental_yield":4.0,"zone":"South Mumbai"},
    "Parel":{"safety":7,"connectivity":8,"pollution":6,"infra":7,"growth":9,"rental_yield":5.0,"zone":"Central Mumbai"},
    "Lower Parel":{"safety":7,"connectivity":9,"pollution":6,"infra":8,"growth":9,"rental_yield":5.2,"zone":"Central Mumbai"},
    "Worli":{"safety":8,"connectivity":8,"pollution":5,"infra":8,"growth":9,"rental_yield":4.8,"zone":"Central Mumbai"},
    "Prabhadevi":{"safety":8,"connectivity":8,"pollution":5,"infra":8,"growth":8,"rental_yield":4.5,"zone":"Central Mumbai"},
    "Dadar West":{"safety":7,"connectivity":9,"pollution":7,"infra":8,"growth":7,"rental_yield":4.8,"zone":"Central Mumbai"},
    "Dadar East":{"safety":7,"connectivity":9,"pollution":7,"infra":8,"growth":7,"rental_yield":4.6,"zone":"Central Mumbai"},
    "Mahim":{"safety":7,"connectivity":8,"pollution":6,"infra":7,"growth":7,"rental_yield":4.5,"zone":"Central Mumbai"},
    "Sion":{"safety":6,"connectivity":9,"pollution":7,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Central Mumbai"},
    "Wadala":{"safety":6,"connectivity":8,"pollution":7,"infra":7,"growth":9,"rental_yield":5.8,"zone":"Central Mumbai"},
    "Byculla":{"safety":5,"connectivity":8,"pollution":8,"infra":6,"growth":9,"rental_yield":5.5,"zone":"Central Mumbai"},
    "Matunga":{"safety":7,"connectivity":9,"pollution":7,"infra":7,"growth":7,"rental_yield":4.8,"zone":"Central Mumbai"},
    "Dharavi":{"safety":4,"connectivity":9,"pollution":8,"infra":5,"growth":10,"rental_yield":7.0,"zone":"Central Mumbai"},
    "Bandra":{"safety":8,"connectivity":9,"pollution":5,"infra":9,"growth":8,"rental_yield":4.2,"zone":"Bandra"},
    "Bandra West":{"safety":9,"connectivity":9,"pollution":4,"infra":9,"growth":8,"rental_yield":4.0,"zone":"Bandra"},
    "Bandra East":{"safety":7,"connectivity":9,"pollution":6,"infra":8,"growth":9,"rental_yield":5.0,"zone":"Bandra"},
    "Bandra Kurla Complex":{"safety":9,"connectivity":9,"pollution":5,"infra":9,"growth":8,"rental_yield":5.5,"zone":"BKC"},
    "Khar":{"safety":8,"connectivity":8,"pollution":5,"infra":8,"growth":7,"rental_yield":4.2,"zone":"Bandra"},
    "Khar West":{"safety":8,"connectivity":8,"pollution":5,"infra":8,"growth":7,"rental_yield":4.2,"zone":"Bandra"},
    "Pali Hill":{"safety":9,"connectivity":7,"pollution":4,"infra":8,"growth":7,"rental_yield":3.5,"zone":"Bandra"},
    "Andheri West":{"safety":7,"connectivity":9,"pollution":6,"infra":8,"growth":7,"rental_yield":5.0,"zone":"Andheri"},
    "Andheri East":{"safety":6,"connectivity":9,"pollution":7,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Andheri"},
    "Versova":{"safety":7,"connectivity":7,"pollution":5,"infra":7,"growth":7,"rental_yield":4.5,"zone":"Andheri"},
    "Juhu":{"safety":8,"connectivity":7,"pollution":4,"infra":8,"growth":7,"rental_yield":4.0,"zone":"Andheri"},
    "Santacruz West":{"safety":7,"connectivity":9,"pollution":6,"infra":8,"growth":7,"rental_yield":4.8,"zone":"Suburban West"},
    "Santacruz East":{"safety":6,"connectivity":9,"pollution":7,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Suburban West"},
    "Ville Parle West":{"safety":7,"connectivity":9,"pollution":6,"infra":8,"growth":7,"rental_yield":4.8,"zone":"Suburban West"},
    "Borivali West":{"safety":7,"connectivity":8,"pollution":5,"infra":7,"growth":7,"rental_yield":5.0,"zone":"Suburban West"},
    "Borivali East":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Suburban West"},
    "Kandivali West":{"safety":7,"connectivity":8,"pollution":5,"infra":7,"growth":7,"rental_yield":5.0,"zone":"Suburban West"},
    "Kandivali East":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Suburban West"},
    "Malad West":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Suburban West"},
    "Malad East":{"safety":6,"connectivity":7,"pollution":6,"infra":6,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Goregaon West":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.3,"zone":"Suburban West"},
    "Goregaon East":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Dahisar East":{"safety":6,"connectivity":7,"pollution":5,"infra":6,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Dahisar West":{"safety":6,"connectivity":7,"pollution":5,"infra":6,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Jogeshwari West":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Jogeshwari East":{"safety":6,"connectivity":8,"pollution":6,"infra":6,"growth":8,"rental_yield":5.5,"zone":"Suburban West"},
    "Powai":{"safety":8,"connectivity":7,"pollution":4,"infra":9,"growth":8,"rental_yield":5.0,"zone":"Central Suburbs"},
    "Ghatkopar East":{"safety":6,"connectivity":9,"pollution":7,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Central Suburbs"},
    "Ghatkopar West":{"safety":7,"connectivity":9,"pollution":6,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Central Suburbs"},
    "Vikhroli":{"safety":6,"connectivity":8,"pollution":7,"infra":6,"growth":8,"rental_yield":5.5,"zone":"Central Suburbs"},
    "Kurla":{"safety":5,"connectivity":9,"pollution":8,"infra":6,"growth":9,"rental_yield":6.0,"zone":"Central Suburbs"},
    "Chembur":{"safety":7,"connectivity":8,"pollution":6,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Central Suburbs"},
    "Mulund West":{"safety":7,"connectivity":7,"pollution":5,"infra":7,"growth":8,"rental_yield":5.0,"zone":"Central Suburbs"},
    "Mulund East":{"safety":7,"connectivity":7,"pollution":5,"infra":7,"growth":8,"rental_yield":5.0,"zone":"Central Suburbs"},
    "Bhandup West":{"safety":6,"connectivity":7,"pollution":6,"infra":6,"growth":8,"rental_yield":5.2,"zone":"Central Suburbs"},
    "Kanjurmarg East":{"safety":7,"connectivity":8,"pollution":5,"infra":7,"growth":9,"rental_yield":5.5,"zone":"Central Suburbs"},
    "Chandivali":{"safety":7,"connectivity":7,"pollution":5,"infra":7,"growth":8,"rental_yield":5.2,"zone":"Central Suburbs"},
    "Thane West":{"safety":7,"connectivity":8,"pollution":6,"infra":8,"growth":9,"rental_yield":5.5,"zone":"Thane"},
    "Thane East":{"safety":6,"connectivity":8,"pollution":6,"infra":7,"growth":9,"rental_yield":5.8,"zone":"Thane"},
    "Ghodbunder Road":{"safety":7,"connectivity":6,"pollution":4,"infra":7,"growth":9,"rental_yield":5.5,"zone":"Thane"},
    "Hiranandani Estates":{"safety":9,"connectivity":6,"pollution":3,"infra":9,"growth":8,"rental_yield":4.8,"zone":"Thane"},
    "Mira Road East":{"safety":6,"connectivity":7,"pollution":5,"infra":6,"growth":9,"rental_yield":6.0,"zone":"MMR Suburbs"},
    "Vashi":{"safety":8,"connectivity":8,"pollution":4,"infra":8,"growth":8,"rental_yield":5.2,"zone":"Navi Mumbai"},
    "Kharghar":{"safety":7,"connectivity":7,"pollution":3,"infra":7,"growth":9,"rental_yield":5.8,"zone":"Navi Mumbai"},
    "Panvel":{"safety":7,"connectivity":7,"pollution":4,"infra":7,"growth":9,"rental_yield":6.0,"zone":"Navi Mumbai"},
    "Nerul":{"safety":8,"connectivity":8,"pollution":3,"infra":8,"growth":8,"rental_yield":5.5,"zone":"Navi Mumbai"},
    "Kamothe":{"safety":7,"connectivity":6,"pollution":3,"infra":6,"growth":9,"rental_yield":6.0,"zone":"Navi Mumbai"},
    "Seawoods":{"safety":8,"connectivity":7,"pollution":3,"infra":7,"growth":8,"rental_yield":5.5,"zone":"Navi Mumbai"},
    "Airoli":{"safety":7,"connectivity":7,"pollution":4,"infra":7,"growth":9,"rental_yield":5.8,"zone":"Navi Mumbai"},
    "Belapur":{"safety":8,"connectivity":7,"pollution":3,"infra":8,"growth":8,"rental_yield":5.2,"zone":"Navi Mumbai"},
    "Ulwe":{"safety":6,"connectivity":6,"pollution":3,"infra":6,"growth":10,"rental_yield":6.5,"zone":"Navi Mumbai"},
    "Vasai West":{"safety":6,"connectivity":6,"pollution":4,"infra":6,"growth":8,"rental_yield":6.5,"zone":"Vasai-Virar"},
    "Virar West":{"safety":6,"connectivity":6,"pollution":3,"infra":5,"growth":8,"rental_yield":7.0,"zone":"Vasai-Virar"},
    "Virar East":{"safety":6,"connectivity":6,"pollution":3,"infra":5,"growth":8,"rental_yield":7.0,"zone":"Vasai-Virar"},
    "Nalasopara East":{"safety":5,"connectivity":6,"pollution":4,"infra":5,"growth":8,"rental_yield":7.0,"zone":"Vasai-Virar"},
    "Nalasopara West":{"safety":5,"connectivity":6,"pollution":4,"infra":5,"growth":8,"rental_yield":7.0,"zone":"Vasai-Virar"},
    "Kalyan West":{"safety":6,"connectivity":7,"pollution":6,"infra":6,"growth":8,"rental_yield":6.0,"zone":"Kalyan-Dombivli"},
    "Kalyan East":{"safety":5,"connectivity":7,"pollution":6,"infra":5,"growth":8,"rental_yield":6.0,"zone":"Kalyan-Dombivli"},
    "Dombivali East":{"safety":6,"connectivity":7,"pollution":5,"infra":6,"growth":8,"rental_yield":6.0,"zone":"Kalyan-Dombivli"},
    "Badlapur East":{"safety":6,"connectivity":6,"pollution":3,"infra":5,"growth":8,"rental_yield":6.5,"zone":"Badlapur"},
    "Badlapur West":{"safety":6,"connectivity":6,"pollution":3,"infra":5,"growth":8,"rental_yield":6.5,"zone":"Badlapur"},
    "Palava":{"safety":8,"connectivity":5,"pollution":2,"infra":8,"growth":10,"rental_yield":6.5,"zone":"Emerging"},
    "_default":{"safety":6,"connectivity":6,"pollution":6,"infra":6,"growth":7,"rental_yield":5.0,"zone":"Mumbai MMR"},
}

INFRA_PROJECTS = {
    "South Mumbai":["Coastal Road Phase 2","Metro Line 3 (Colaba-SEEPZ)"],
    "BKC":["Metro Line 3","BKC-Worli Connector"],
    "Central Mumbai":["Metro Line 3","Coastal Road"],
    "Bandra":["Metro Line 2A","Coastal Road"],
    "Andheri":["Metro Line 1 & 2A","JVLR widening"],
    "Suburban West":["Metro Line 2A & 7","Western Expressway upgrade"],
    "Central Suburbs":["Metro Line 4","Eastern Freeway extension"],
    "Thane":["Metro Line 4 & 5","Thane Ring Road"],
    "Navi Mumbai":["Navi Mumbai Metro","NMIA Airport (2025)"],
    "MMR Suburbs":["Virar-Alibaug Corridor","Mira Road Metro"],
    "Kalyan-Dombivli":["Metro Line 5","KDMC Smart City Phase 2"],
    "Vasai-Virar":["Virar-Alibaug Multimodal Corridor"],
    "Badlapur":["CSMT-Panvel AC Locals","Kalyan Ring Road"],
    "Emerging":["Navi Mumbai Airport","Palava Smart City"],
    "Mumbai MMR":["Mumbai Urban Transport Project"],
    "_default":["Mumbai Urban Transport Project"],
}

def get_intel(r): return REGION_DATA.get(r, REGION_DATA["_default"])
def encode(col, val):
    le = encoders[col]
    return int(le.transform([val])[0]) if val in le.classes_ else 0

def predict_price(p):
    f = np.array([[int(p["bhk"]),encode("type",p["type"]),int(p["area"]),encode("region",p["region"]),encode("status",p["status"]),encode("age",p["age"])]])
    return round(float(model.predict(f)[0]), 2)

def fair_price_label(pred, listed):
    if listed is None: return None
    d = ((listed-pred)/pred)*100
    if d>10: return {"label":"Overpriced","color":"red","diff_pct":round(d,1)}
    if d<-10: return {"label":"Underpriced","color":"green","diff_pct":round(d,1)}
    return {"label":"Fair Price","color":"blue","diff_pct":round(d,1)}

def emi(price, down=20, rate=8.5, years=20):
    loan=price*(1-down/100)*100_000; r=rate/(12*100); n=years*12
    if r==0: return round(loan/n)
    return round(loan*r*(1+r)**n/((1+r)**n-1))

def liveability(intel):
    s,c,p_inv,i = intel["safety"],intel["connectivity"],11-intel["pollution"],intel["infra"]
    score=round(s*.3+c*.3+p_inv*.2+i*.2,1)
    label="Excellent" if score>=8 else "Very Good" if score>=7 else "Good" if score>=6 else "Average" if score>=5 else "Below Average"
    return {"score":score,"label":label,"safety":s,"connectivity":c,"air_quality":p_inv,"infrastructure":i,"zone":intel.get("zone","Mumbai MMR")}

def investment(intel, price):
    r,g=intel["rental_yield"],intel["growth"]
    risk=max(1,min(10,round((10-(g+r)/2)/2+3,1)))
    return {"rental_yield_score":round(min(10,r*1.2),1),"appreciation_score":round(g,1),"overall_investment_score":round(r*.4+g*.6,1),"rental_yield_pct":r,"risk_score":round(risk,1),"risk_label":"Low" if risk<4 else "Medium" if risk<7 else "High"}

def future_prices(price, intel):
    g=intel["growth"]/100; zone=intel.get("zone","_default")
    return {"year_1":round(price*(1+g),2),"year_2":round(price*(1+g)**2,2),"year_3":round(price*(1+g)**3,2),"growth_rate_pct":intel["growth"],"infrastructure_projects":INFRA_PROJECTS.get(zone,INFRA_PROJECTS["_default"])}

def full_analysis(p):
    price=predict_price(p); intel=get_intel(p["region"])
    return price,intel,liveability(intel),investment(intel,price),future_prices(price,intel)

# ─── Auth Routes ──────────────────────────────────────────────────────────────
@app.route("/auth/register", methods=["POST"])
def register():
    d=request.get_json() or {}
    name=(d.get("name","") or "").strip(); email=(d.get("email","") or "").strip().lower()
    phone=(d.get("phone","") or "").strip(); pwd=d.get("password","") or ""
    if not name or not email or not pwd: return jsonify({"error":"Name, email and password required"}),400
    if len(pwd)<6: return jsonify({"error":"Password must be at least 6 characters"}),400
    try:
        uid=str(uuid.uuid4())
        with get_db() as conn:
            conn.execute("INSERT INTO users (id,name,email,phone,password,created_at) VALUES (?,?,?,?,?,?)",
                (uid,name,email,phone,hash_password(pwd),datetime.utcnow().isoformat()))
        token=create_session(uid)
        return jsonify({"ok":True,"token":token,"user":{"id":uid,"name":name,"email":email}}),201
    except sqlite3.IntegrityError:
        return jsonify({"error":"Email already registered"}),409
    except Exception as e:
        return jsonify({"error":str(e)}),500

@app.route("/auth/login", methods=["POST"])
def login():
    d=request.get_json() or {}
    email=(d.get("email","") or "").strip().lower(); pwd=d.get("password","") or ""
    if not email or not pwd: return jsonify({"error":"Email and password required"}),400
    with get_db() as conn:
        user=conn.execute("SELECT * FROM users WHERE email=?",(email,)).fetchone()
    if not user or not verify_password(pwd,user["password"]): return jsonify({"error":"Invalid email or password"}),401
    with get_db() as conn:
        conn.execute("UPDATE users SET last_login=? WHERE id=?",(datetime.utcnow().isoformat(),user["id"]))
    token=create_session(user["id"])
    return jsonify({"ok":True,"token":token,"user":{"id":user["id"],"name":user["name"],"email":user["email"]}})

@app.route("/auth/logout", methods=["POST"])
def logout():
    token=request.headers.get("X-Auth-Token")
    if token:
        with get_db() as conn: conn.execute("DELETE FROM sessions WHERE token=?",(token,))
    return jsonify({"ok":True})

@app.route("/auth/profile", methods=["GET"])
def profile():
    user=current_user()
    if not user: return jsonify({"error":"Not authenticated"}),401
    with get_db() as conn:
        searches=conn.execute("SELECT type,inputs,result_price,region,created_at FROM searches WHERE user_id=? ORDER BY created_at DESC LIMIT 20",(user["id"],)).fetchall()
    return jsonify({"user":{"id":user["id"],"name":user["name"],"email":user["email"],"phone":user.get("phone"),"created_at":user["created_at"],"last_login":user.get("last_login")},"recent_searches":[dict(s) for s in searches]})

@app.route("/admin/users", methods=["GET"])
def admin_users():
    if request.headers.get("X-Admin-Secret") != os.environ.get("ADMIN_SECRET","realytics-admin-2025"):
        return jsonify({"error":"Forbidden"}),403
    with get_db() as conn:
        users=conn.execute("SELECT id,name,email,phone,created_at,last_login FROM users ORDER BY created_at DESC").fetchall()
    return jsonify({"users":[dict(u) for u in users],"total":len(users)})

@app.route("/admin/searches", methods=["GET"])
def admin_searches():
    if request.headers.get("X-Admin-Secret") != os.environ.get("ADMIN_SECRET","realytics-admin-2025"):
        return jsonify({"error":"Forbidden"}),403
    with get_db() as conn:
        rows=conn.execute("SELECT s.id,s.type,s.inputs,s.result_price,s.region,s.created_at,u.name as user_name,u.email as user_email,s.guest_id FROM searches s LEFT JOIN users u ON u.id=s.user_id ORDER BY s.created_at DESC LIMIT 200").fetchall()
    return jsonify({"searches":[dict(r) for r in rows],"total":len(rows)})

# ─── Core Routes ──────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def home(): return jsonify({"status":"Mumbai House Price API v3","version":"3.0"})

@app.route("/meta", methods=["GET"])
def get_meta(): return jsonify(meta)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data=request.get_json()
        for f in ["bhk","type","area","region","status","age"]:
            if f not in data: return jsonify({"error":f"Missing field: {f}"}),400
        price,intel,live,inv,fut=full_analysis(data)
        listed=data.get("listed_price")
        log_search("predict",data,price,data.get("region"))
        return jsonify({"predicted_price_lakhs":price,"predicted_price_cr":round(price/100,3),"emi_per_month":emi(price),"fair_price":fair_price_label(price,listed),"liveability":live,"investment":inv,"future_prices":fut,"inputs":data})
    except Exception as e: return jsonify({"error":str(e)}),500

@app.route("/compare", methods=["POST"])
def compare():
    try:
        data=request.get_json(); props=data.get("properties",[])
        if len(props)!=2: return jsonify({"error":"Provide exactly 2 properties"}),400
        results=[]
        for p in props:
            price,intel,live,inv,fut=full_analysis(p)
            results.append({"label":p.get("label",p["region"]),"predicted_price_lakhs":price,"predicted_price_cr":round(price/100,3),"emi_per_month":emi(price),"liveability_score":live["score"],"liveability_label":live["label"],"investment_score":inv["overall_investment_score"],"rental_yield":inv["rental_yield_pct"],"appreciation_score":inv["appreciation_score"],"risk_label":inv["risk_label"],"year_3_price_lakhs":fut["year_3"],"growth_rate_pct":fut["growth_rate_pct"],"zone":intel.get("zone","Mumbai MMR"),"infrastructure":fut["infrastructure_projects"],"inputs":p})
        w=0 if results[0]["investment_score"]>=results[1]["investment_score"] else 1
        results[w]["recommended"]=True; results[1-w]["recommended"]=False
        log_search("compare",props,results[w]["predicted_price_lakhs"],props[0].get("region"))
        return jsonify({"properties":results})
    except Exception as e: return jsonify({"error":str(e)}),500

if __name__=="__main__":
    print("Starting Realytics API v3 on http://localhost:5000")
    app.run(debug=True, port=5000)
