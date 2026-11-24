def _serve_asset(kind, ver):
    a = ASSETS[kind]
    if ver != a["ver"]:
        abort(404)
    etag = '"%s"' % a["ver"]
    if request.headers.get("If-None-Match") == etag:
        resp = app.response_class(status=304)
    else:
        gz = "gzip" in request.headers.get("Accept-Encoding", "").lower()
        resp = app.response_class(a["gz"] if gz else a["raw"], mimetype=a["mimetype"])
        if gz:
            resp.headers["Content-Encoding"] = "gzip"
    resp.headers.update({"ETag": etag, "Cache-Control": "public, max-age=31536000, immutable", "Vary": "Accept-Encoding"})
    return resp


@app.route("/assets/app.<ver>.css")
def asset_css(ver):
    return _serve_asset("css", ver)


@app.route("/assets/app.<ver>.js")
def asset_js(ver):
    return _serve_asset("js", ver)


@app.context_processor
def inject_globals():
    return {"disclaimer": DISCLAIMER, "ai_ready": ai_configured(), "asset_css": ASSET_URLS["css"], "asset_js": ASSET_URLS["js"]}


# ---------- Public pages ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/health")
def health():
    return {"status": "ok"}


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        confirm = request.form.get("confirm", "")
        agree = request.form.get("agree")
        error = None
        if len(name) < 2:
            error = "Enter your full name."
        elif not EMAIL_RE.match(email):
            error = "Enter a valid email address."
        elif len(pw) < 8:
            error = "Password must be at least 8 characters."
        elif pw != confirm:
            error = "Passwords do not match."
        elif not agree:
            error = "Please accept the disclaimer and terms to continue."
        if not error and not ip_rate_ok("signup", 10, 3600):
            error = "Too many sign-ups from this network. Please try again later."
        if not error:
            try:
                query("INSERT INTO users (name, email, password_hash) VALUES (?,?,?)",
                      (name, email, generate_password_hash(pw)), commit=True)
                new = query("SELECT id FROM users WHERE email = ?", (email,), one=True)
                audit("signup", user_id=new["id"] if new else None)
                flash("Account created. Please log in.", "success")
                return redirect(url_for("login"))
            except IntegrityError:
                get_db().rollback()
                error = "An account with this email already exists."
        flash(error, "error")
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        pw = request.form.get("password", "")
        if not ip_rate_ok("login", 10, 900):
            flash("Too many log-in attempts. Please wait 15 minutes and try again.", "error")
            return render_template("login.html"), 429
        user = query("SELECT * FROM users WHERE email = ?", (email,), one=True)
        if user and check_password_hash(user["password_hash"], pw):
            session.clear()
            session["user_id"] = user["id"]
            audit("login", user_id=user["id"])
            return redirect(safe_next(request.args.get("next")) or url_for("dashboard"))
        if user:
            audit("login_failed", user_id=user["id"])
        flash("Incorrect email or password.", "error")
    return render_template("login.html")


@app.route("/logout", methods=["POST"])
def logout():
    if g.user:
        audit("logout")
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


# ---------- App pages (login required) ----------
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", features=feature_list(), live=live_status(), labels=STATUS_LABEL,
                           consents=get_consents(), purposes=CONSENT_PURPOSES,
                           device_types=DEVICE_TYPES, reg_status=REG_STATUS,
                           reading_metrics=READING_METRICS)


# ---------- Digital Patient Twin page (3D hologram, built into app.py: no separate template file needed) ----------
TWIN_HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Digital Patient Twin · MediBridge AI</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Figtree:wght@400;600&family=Fraunces:opsz,wght@9..144,500&display=swap" rel="stylesheet">
<style>
:root{--bg:#0d1318;--panel:#141c23;--line:#25313b;--ink:#e9eff3;--mute:#8d9ead;--heart:#ea5560;--teal:#5ff3ff;color-scheme:dark}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:400 15px/1.45 Figtree,system-ui,sans-serif}
.app{display:grid;grid-template-columns:1fr 360px;height:100%}
#stage{position:relative;min-height:0;overflow:hidden;touch-action:none;cursor:grab;background:radial-gradient(ellipse at 50% 42%,#0a2b38 0%,#050c11 75%)}
#stage::after{content:"";position:absolute;inset:0;pointer-events:none;background:repeating-linear-gradient(0deg,rgba(95,243,255,.04) 0 1px,transparent 1px 3px);box-shadow:inset 0 0 120px #000c}
#stage:active{cursor:grabbing}
#stage canvas{display:block;width:100%;height:100%}
.bar{position:absolute;left:16px;top:calc(14px + env(safe-area-inset-top,0px));right:16px;display:flex;justify-content:space-between;align-items:flex-start;pointer-events:none}
.bar>*{pointer-events:auto}
.back{color:var(--mute);text-decoration:none;font-size:14px}
.back:hover{color:var(--ink)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:999px;overflow:hidden;background:#0d1318aa;backdrop-filter:blur(6px)}
.seg button{all:unset;cursor:pointer;padding:6px 13px;font-size:13px;color:var(--mute)}
.seg button[aria-pressed=true]{background:var(--ink);color:#0d1318;font-weight:600}
.seg button:focus-visible,.back:focus-visible{outline:2px solid var(--teal);outline-offset:2px}
.hint{position:absolute;left:0;right:0;bottom:12px;text-align:center;font-size:12px;color:var(--mute)}
aside{background:var(--panel);border-left:1px solid var(--line);overflow:auto;padding:calc(18px + env(safe-area-inset-top,0px)) 20px 24px;display:flex;flex-direction:column;gap:14px}
h1{font:500 26px/1.15 Fraunces,Georgia,serif;margin:0}
.sub{color:var(--mute);font-size:13px;display:flex;gap:8px;align-items:center}
.dot{width:8px;height:8px;border-radius:50%;background:#e0a23a}
.dot.on{background:#4fd18b;box-shadow:0 0 0 4px #4fd18b22}
.row{display:grid;grid-template-columns:1fr 84px;gap:4px 10px;padding:12px 0;border-top:1px solid var(--line);align-items:center}
.row b{font-weight:600}
.val{font:500 30px/1 Fraunces,Georgia,serif;font-variant-numeric:tabular-nums}
.val small{font:400 12px Figtree,sans-serif;color:var(--mute);margin-left:5px}
.meta{font-size:12px;color:var(--mute)}
.row canvas{grid-column:2;grid-row:1/3;width:84px;height:38px}
.note{font-size:12px;color:var(--mute);border-top:1px solid var(--line);padding-top:12px;margin-top:auto}
.empty{background:#1c2630;border-radius:8px;padding:10px 12px;font-size:13px;color:var(--mute);display:none}
@media(max-width:820px){.app{grid-template-columns:1fr;grid-template-rows:56vh 1fr}aside{border-left:0;border-top:1px solid var(--line)}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
</head>
<body>
<div class="app">
  <div id="stage" aria-label="Interactive 3D model of the body">
    <div class="bar">
      <a class="back" href="/dashboard">‹ Dashboard</a>
      <div class="seg" id="view" role="group" aria-label="Layers">
        <button data-v="skin" aria-pressed="false">Body</button>
        <button data-v="muscle" aria-pressed="false">Muscles</button>
        <button data-v="organs" aria-pressed="true">Organs</button>
      </div>
    </div>
    <div class="hint">Drag to rotate · scroll or pinch to zoom</div>
  </div>
  <aside>
    <div>
      <h1>Digital Patient Twin</h1>
      <div class="sub"><span class="dot" id="dot"></span><span id="live">Connecting…</span><span>·</span><span>{{ patient_id }}</span></div>
    </div>
    <div class="seg" id="range" role="group" aria-label="History period" style="align-self:flex-start">
      <button data-d="1" aria-pressed="true">24 hours</button>
      <button data-d="7" aria-pressed="false">7 days</button>
      <button data-d="30" aria-pressed="false">30 days</button>
    </div>
    <div class="empty" id="empty">No readings yet. The model shows a resting rhythm until you add a reading or connect a device.</div>
    <div id="rows"></div>
    <p class="note">Educational prototype. The model shows your own readings and is not a diagnosis. Talk to a clinician about any health concern.</p>
  </aside>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
<script>
(function () {
  const $ = s => document.querySelector(s);
  const META = {heart_rate:["Heart rate","bpm"], spo2:["Oxygen (SpO2)","%"], temperature:["Temperature","°C"], resp_rate:["Breathing","breaths/min"], bp:["Blood pressure","mmHg"]};
  let state = {latest:{}, history:{}}, days = 1;
  const cur = {hr:72, rr:15}, tgt = {hr:72, rr:15};

  /* ---------- scene ---------- */
  const stage = $("#stage");
  const R = new THREE.WebGLRenderer({antialias:true, alpha:true});
  R.setPixelRatio(Math.min(devicePixelRatio, 2));
  R.outputEncoding = THREE.sRGBEncoding;
  R.toneMapping = THREE.ACESFilmicToneMapping;
  R.toneMappingExposure = 1.15;
  stage.appendChild(R.domElement);
  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(30, 1, 0.1, 50);

  scene.add(new THREE.HemisphereLight(0xcfe3ff, 0x20160f, 0.75));
  const key = new THREE.DirectionalLight(0xfff1e6, 1.1); key.position.set(2.5, 3.5, 3); scene.add(key);
  const rim = new THREE.PointLight(0x7fd1c3, 1.6, 8); rim.position.set(-2.2, 2.2, -2); scene.add(rim);
  const fill = new THREE.PointLight(0xffc9b0, 0.5, 8); fill.position.set(1.5, 0.6, 2.5); scene.add(fill);

  const body = new THREE.Group(); scene.add(body);

  // holographic projector: base, floor rings, light cone, scan rings, rising dust
  const CY = "#5ff3ff";
  const disc = new THREE.Mesh(new THREE.CylinderGeometry(.62, .7, .07, 64), new THREE.MeshStandardMaterial({color:"#0b1820", metalness:.9, roughness:.3}));
  disc.position.y = -.035; scene.add(disc);
  const emit = new THREE.Mesh(new THREE.TorusGeometry(.56, .012, 12, 96), new THREE.MeshBasicMaterial({color:CY}));
  emit.rotation.x = Math.PI/2; emit.position.y = .002; scene.add(emit);
  const c = document.createElement("canvas"); c.width = c.height = 512;
  const cx = c.getContext("2d"), gr = cx.createRadialGradient(256,256,0,256,256,256);
  gr.addColorStop(0,"rgba(95,243,255,.5)"); gr.addColorStop(.6,"rgba(95,243,255,.12)"); gr.addColorStop(1,"rgba(95,243,255,0)");
  cx.fillStyle = gr; cx.fillRect(0,0,512,512); cx.strokeStyle = "rgba(95,243,255,.55)"; cx.lineWidth = 2;
  [90,130,170,215].forEach(r => { cx.beginPath(); cx.arc(256,256,r,0,7); cx.stroke(); });
  for (let i = 0; i < 72; i++) { const a = i*Math.PI/36, l = i%6 ? 228 : 244; cx.beginPath(); cx.moveTo(256+Math.cos(a)*215, 256+Math.sin(a)*215); cx.lineTo(256+Math.cos(a)*l, 256+Math.sin(a)*l); cx.stroke(); }
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(2.6,2.6), new THREE.MeshBasicMaterial({map:new THREE.CanvasTexture(c), transparent:true, depthWrite:false, blending:THREE.AdditiveBlending}));
  floor.rotation.x = -Math.PI/2; floor.position.y = .004; scene.add(floor);
  const cone = new THREE.Mesh(new THREE.CylinderGeometry(.55, .62, 2.1, 48, 1, true), new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, side:THREE.DoubleSide, blending:THREE.AdditiveBlending,
    vertexShader:"varying float h;void main(){h=position.y/2.1+.5;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}",
    fragmentShader:"varying float h;void main(){gl_FragColor=vec4(.37,.95,1.,.17*(1.-h)*(1.-h));}"}));
  cone.position.y = 1.05; scene.add(cone);
  const rings = [0,1,2].map(i => { const r = new THREE.Mesh(new THREE.TorusGeometry(.52 + .03*i, .004, 6, 96), new THREE.MeshBasicMaterial({color:CY, transparent:true, opacity:.4, blending:THREE.AdditiveBlending, depthWrite:false})); r.rotation.x = Math.PI/2; scene.add(r); return r; });
  const N = 260, pa = new Float32Array(N*3), pv = [];
  for (let i = 0; i < N; i++) { const a = Math.random()*6.283, r = .15 + Math.random()*.5; pa[i*3] = Math.cos(a)*r; pa[i*3+1] = Math.random()*2; pa[i*3+2] = Math.sin(a)*r; pv.push(.04 + Math.random()*.08); }
  const pg = new THREE.BufferGeometry(); pg.setAttribute("position", new THREE.BufferAttribute(pa, 3));
  scene.add(new THREE.Points(pg, new THREE.PointsMaterial({color:"#9ff8ff", size:.012, transparent:true, opacity:.75, blending:THREE.AdditiveBlending, depthWrite:false})));


  // translucent skin with fresnel rim
  const skinMat = new THREE.ShaderMaterial({
    transparent:true, depthWrite:false, side:THREE.DoubleSide, blending:THREE.AdditiveBlending,
    uniforms:{c:{value:new THREE.Color("#4fe3ff")}, k:{value:.1}, uT:{value:0}},
    vertexShader:"varying vec3 n;varying vec3 v;varying vec3 w;void main(){vec4 mv=modelViewMatrix*vec4(position,1.);w=(modelMatrix*vec4(position,1.)).xyz;n=normalize(normalMatrix*normal);v=normalize(-mv.xyz);gl_Position=projectionMatrix*mv;}",
    fragmentShader:"uniform vec3 c;uniform float k;uniform float uT;varying vec3 n;varying vec3 v;varying vec3 w;void main(){float f=pow(1.-abs(dot(n,v)),2.);float sc=.5+.5*sin(w.y*230.-uT*3.);float bm=exp(-pow((w.y-(mod(uT*.35,2.3)-.15))/.035,2.));float fl=.93+.07*sin(uT*37.)*sin(uT*11.);vec3 col=c*(.25+f*1.7)+c*sc*.3+vec3(.8,1.,1.)*bm;float a=(k+f*.85+sc*.1+bm*.7)*fl;gl_FragColor=vec4(col*a,a);}"
  });
  const skin = new THREE.Group(); body.add(skin);
  const add = (geo, x, y, z, sx, sy, sz, rz) => { const m = new THREE.Mesh(geo, skinMat); m.position.set(x,y,z); m.scale.set(sx||1,sy||1,sz||1); m.rotation.z = rz||0; skin.add(m); return m; };
  const S = (r) => new THREE.SphereGeometry(r, 32, 24);
  const C = (rt, rb, h) => new THREE.CylinderGeometry(rt, rb, h, 28, 1);

  // torso (smooth lathe)
  const prof = new THREE.SplineCurve([[0,.90],[.14,.91],[.185,.98],[.175,1.10],[.15,1.21],[.16,1.33],[.2,1.42],[.2,1.48],[.12,1.54],[.06,1.58],[0,1.6]].map(p => new THREE.Vector2(p[0], p[1]))).getPoints(48).map(p => new THREE.Vector2(p.x, p.y));
  const torso = new THREE.Mesh(new THREE.LatheGeometry(prof.map(p=>new THREE.Vector2(Math.max(p.x,0), p.y)), 48), skinMat);
  torso.scale.set(1.4, 1, 0.74); skin.add(torso);
  add(S(.11), 0, 1.745, 0, .88, 1.12, 1);            // head
  add(S(.075), 0, 1.68, .03, .95, .9, 1);            // jaw
  add(C(.048,.055,.14), 0, 1.6, 0);                  // neck
  add(S(.17), 0, .92, 0, 1.1, .6, .8);               // pelvis
  [-1,1].forEach(s => {
    add(S(.06), s*.29, 1.47, 0);                     // shoulder
    add(C(.048,.055,.3), s*.325, 1.32, 0, 1,1,1, s*.12);   // upper arm
    add(S(.045), s*.355, 1.17, 0);                   // elbow
    add(C(.034,.045,.3), s*.385, 1.02, 0, 1,1,1, s*.06);  // forearm
    add(S(.04), s*.4, .85, 0, .8, 1.2, .5);          // hand
    add(C(.06,.088,.46), s*.1, .72, 0, 1,1,1, s*-.02);    // thigh
    add(S(.06), s*.1, .49, 0);                       // knee
    add(C(.038,.058,.46), s*.1, .26, 0);             // calf
    add(S(.05), s*.1, .045, .06, .9, .5, 1.8);       // foot
  });

  /* ---------- organs ---------- */
  const organs = new THREE.Group(); body.add(organs);
  const mat = (col, o) => new THREE.MeshPhysicalMaterial(Object.assign({color:col, roughness:.45, clearcoat:.6, clearcoatRoughness:.35}, o||{}));
  const place = (m, x, y, z, sx, sy, sz) => { m.position.set(x,y,z); m.scale.set(sx||1, sy||1, sz||1); organs.add(m); return m; };

  // anatomical heart: a puffy heart-outline surface, apex pointing to the patient's left
  const heart = new THREE.Group(); heart.position.set(.045, 1.31, .035); heart.rotation.z = .5; organs.add(heart);
  const hm = mat("#c3303b", {roughness:.3, clearcoat:1, clearcoatRoughness:.2, emissive:"#4a0a10", emissiveIntensity:.45});
  const hg = new THREE.SphereGeometry(1, 64, 48), hp = hg.attributes.position;
  for (let i = 0; i < hp.count; i++) {
    const px = hp.getX(i), py = hp.getY(i), pz = hp.getZ(i), ph = Math.acos(Math.max(-1, Math.min(1, py))), t = Math.atan2(pz, px) + Math.PI, sp = Math.sin(ph);
    const hx = 16 * Math.pow(Math.sin(t), 3), hy = 13 * Math.cos(t) - 5 * Math.cos(2*t) - 2 * Math.cos(3*t) - Math.cos(4*t);
    hp.setXYZ(i, hx * sp * .0026, (hy + 2) * sp * .0026, Math.cos(ph) * 11 * .0026 * (.55 + .45 * sp));
  }
  hg.computeVertexNormals(); heart.add(new THREE.Mesh(hg, hm));

  // vessels: tubes with a pulse of light that travels from the heart on every beat
  const pulseU = {uPh:{value:0}};
  const vmat = col => new THREE.ShaderMaterial({
    uniforms:Object.assign({uC:{value:new THREE.Color(col)}}, pulseU),
    vertexShader:"varying vec2 vUv;varying vec3 vN;void main(){vUv=uv;vN=normalize(normalMatrix*normal);gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}",
    fragmentShader:"uniform vec3 uC;uniform float uPh;varying vec2 vUv;varying vec3 vN;void main(){float l=.5+.5*dot(normalize(vN),normalize(vec3(.3,.6,.7)));float p=exp(-pow((vUv.x-(uPh*1.5-.15))/.09,2.));gl_FragColor=vec4(uC*l+vec3(1.,.55,.5)*p*.9,1.);}"
  });
  const artM = vmat("#b3262e"), veinM = vmat("#4560b5");
  const vessel = (pts, m, r) => organs.add(new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(p[0], p[1], p[2]))), 60, r, 10, false), m));
  vessel([[.05,1.37,.04],[.045,1.46,.035],[.01,1.5,.0],[-.015,1.45,-.03],[0,1.3,-.055],[0,1.05,-.045],[0,.95,-.03]], artM, .012);   // aorta
  [-1,1].forEach(s => {
    vessel([[0,.95,-.03],[s*.05,.93,-.015],[s*.1,.7,0],[s*.1,.3,.01],[s*.1,.08,.02]], artM, .008);                  // leg arteries
    vessel([[.01,1.5,.0],[s*.2,1.47,.0],[s*.33,1.3,0],[s*.38,1.05,0],[s*.4,.88,0]], artM, .006);                     // arm arteries
    vessel([[.01,1.5,.0],[s*.025,1.62,.0],[s*.035,1.73,.0]], artM, .006);                                            // carotids
    vessel([[.045,1.37,.05],[.02,1.4,.045],[s*.05,1.39,.03],[s*.1,1.37,.0]], veinM, .008);                           // pulmonary vessels
  });
  vessel([[-.02,1.28,.03],[-.035,1.1,-.03],[-.03,.95,-.025]], veinM, .012);                                          // lower vena cava
  vessel([[-.02,1.52,.0],[-.02,1.44,.01],[-.02,1.31,.035]], veinM, .01);                                             // upper vena cava

  const tube = (pts, r, m, seg) => { const t = new THREE.Mesh(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(p => new THREE.Vector3(p[0], p[1], p[2]))), seg || 40, r, 10, false), m); organs.add(t); return t; };
  const lungs = new THREE.Group(); organs.add(lungs);
  const lm = mat("#e79aa4", {roughness:.6, clearcoat:.3, transparent:true, opacity:.93});
  // right lung: 3 lobes. left lung: 2 lobes, narrower (room for the heart)
  [[-.105,1.40,0,.07,.085,.065,0],[-.118,1.30,.02,.082,.045,.07,0],[-.102,1.2,-.012,.085,.085,.08,0],[.112,1.38,0,.064,.1,.064,.12],[.104,1.2,-.01,.078,.09,.08,0]]
    .forEach(a => { const l = new THREE.Mesh(S(1), lm); l.position.set(a[0], a[1], a[2]); l.scale.set(a[3], a[4], a[5]); l.rotation.z = a[6]; lungs.add(l); });
  const cart = mat("#e3d2c2", {roughness:.5});
  tube([[0,1.6,-.015],[0,1.44,-.015]], .013, cart, 8);                                   // trachea
  for (let i = 0; i < 9; i++) { const r = new THREE.Mesh(new THREE.TorusGeometry(.015, .0025, 6, 20), cart); r.rotation.x = Math.PI/2; r.position.set(0, 1.455 + i*.016, -.015); organs.add(r); }
  [-1,1].forEach(s => { tube([[0,1.44,-.015],[s*.04,1.4,-.012],[s*.08,1.33,-.005],[s*.09,1.25,-.01]], .008, cart); tube([[s*.05,1.39,-.012],[s*.1,1.41,-.005],[s*.12,1.44,0]], .005, cart); });
  const dia = new THREE.Mesh(new THREE.SphereGeometry(1, 32, 12, 0, Math.PI*2, 0, Math.PI/2), mat("#b0675c", {transparent:true, opacity:.6, side:THREE.DoubleSide}));
  place(dia, 0, 1.13, 0, .19, .07, .115);                                                 // diaphragm dome
  tube([[0,1.55,-.022],[.008,1.35,-.02],[.02,1.2,-.005]], .008, mat("#d99a8f"));         // esophagus
  place(new THREE.Mesh(S(1), mat("#7a2f2a", {roughness:.5, clearcoat:.5})), -.075, 1.09, .03, .09, .052, .072);   // liver, right lobe
  place(new THREE.Mesh(S(1), mat("#7a2f2a", {roughness:.5, clearcoat:.5})), .0, 1.1, .048, .06, .033, .048);      // liver, left lobe
  place(new THREE.Mesh(S(1), mat("#5f8b4a")), -.04, 1.04, .07, .014, .02, .014);                                  // gallbladder
  const sm = mat("#cf8576", {roughness:.5, clearcoat:.5});
  tube([[.02,1.2,-.005],[.05,1.15,.02],[.095,1.09,.04],[.09,1.02,.045],[.04,.99,.04]], .027, sm, 40);              // stomach (J shape)
  place(new THREE.Mesh(S(1), sm), .07, 1.15, .02, .042, .04, .04);                                                 // fundus
  place(new THREE.Mesh(S(1), mat("#6d2c4a")), .13, 1.09, -.04, .02, .035, .016);                                   // spleen
  place(new THREE.Mesh(S(1), mat("#e3b891")), .03, 1.02, -.005, .075, .013, .016).rotation.z = .12;                 // pancreas
  [-1,1].forEach(s => { place(new THREE.Mesh(S(1), mat("#7b3a3a", {roughness:.45, clearcoat:.5})), s*.07, 1.0, -.06, .02, .04, .02); place(new THREE.Mesh(S(1), mat("#d7a64a")), s*.07, 1.045, -.06, .012, .008, .012); });   // kidneys + adrenals
  const pts = []; for (let k = 0; k <= 46; k++) pts.push([Math.sin(k*.95)*.085, .87 + k/46*.1 + .006*Math.sin(k*2.3), .045 + .028*Math.cos(k*1.3)]);
  tube(pts, .0105, mat("#e8a688", {roughness:.5, clearcoat:.4}), 360);                                             // small intestine coils
  tube([[-.1,.88,.03],[-.115,1.0,.03],[-.11,1.03,.04],[0,1.025,.045],[.11,1.03,.04],[.115,.98,.03],[.1,.88,.03],[.06,.83,.03],[.0,.82,.01]], .015, mat("#c88a70", {roughness:.55}), 120);   // large intestine
  place(new THREE.Mesh(S(1), mat("#d9b7a0", {transparent:true, opacity:.85})), 0, .84, .05, .032, .03, .03);       // bladder


  const brain = new THREE.Mesh(S(.085), mat("#e9aab2", {roughness:.7}));
  const bp = brain.geometry.attributes.position, v = new THREE.Vector3();
  for (let i = 0; i < bp.count; i++) { v.fromBufferAttribute(bp, i); const d = 1 + .045 * Math.sin(v.x*70) * Math.sin(v.y*70 + v.z*40); bp.setXYZ(i, v.x*d, v.y*d, v.z*d); }
  brain.geometry.computeVertexNormals(); place(brain, 0, 1.76, 0, 1, .85, 1.12);



  const bone = mat("#ece3d2", {roughness:.55, clearcoat:.2});
  for (let i = 0; i < 8; i++) {
    const g = new THREE.Group(), m = new THREE.Mesh(new THREE.TorusGeometry(.13 + (i<5?i:9-i)*.006, .0055, 8, 40, Math.PI*1.5), bone);
    m.rotation.z = Math.PI*1.75; g.add(m); g.rotation.x = Math.PI/2; g.scale.set(1.35, .8, 1); g.position.set(0, 1.45 - i*.036, 0); organs.add(g);
  }
  place(new THREE.Mesh(new THREE.BoxGeometry(.028,.2,.01), bone), 0, 1.35, .105);                      // sternum
  for (let i = 0; i < 16; i++) place(new THREE.Mesh(C(.017,.017,.025), bone), 0, 1.5 - i*.036 + (i>11?-.02:0), -.1);  // spine
  [-1,1].forEach(s => { place(new THREE.Mesh(C(.012,.012,.17), bone), s*.15, 1.5, .0, 1,1,1).rotation.z = Math.PI/2.2; });  // collarbones


  /* ---------- muscles: spindle-shaped bellies with fibre grain and pale tendons ---------- */
  const muscles = new THREE.Group(); body.add(muscles); muscles.visible = false;
  const fibre = (() => { const k = document.createElement("canvas"); k.width = 256; k.height = 4; const g = k.getContext("2d");
    for (let x = 0; x < 256; x++) { const v = 150 + Math.random()*70 | 0; g.fillStyle = "rgb(" + v + "," + (v*.36|0) + "," + (v*.34|0) + ")"; g.fillRect(x, 0, 1, 4); }
    const t = new THREE.CanvasTexture(k); t.wrapS = THREE.RepeatWrapping; t.repeat.set(2, 1); t.encoding = THREE.sRGBEncoding; return t; })();
  const mMat = new THREE.MeshPhysicalMaterial({map:fibre, vertexColors:true, roughness:.5, clearcoat:.4, clearcoatRoughness:.45});
  const tendon = new THREE.Color("#e9dccb"), white = new THREE.Color("#ffffff");
  function muscle(p0, p1, r, fx, fz) {
    const a = new THREE.Vector3(...p0), b = new THREE.Vector3(...p1), len = a.distanceTo(b);
    const g = new THREE.SphereGeometry(1, 20, 16), P = g.attributes.position, col = [];
    for (let i = 0; i < P.count; i++) {
      const x = P.getX(i), y = P.getY(i), z = P.getZ(i), h = Math.hypot(x, z) || 1, prof = 1 - Math.pow(Math.abs(y), 1.7);
      P.setXYZ(i, x / h * r * (fx || 1) * prof * (h < 1e-4 ? 0 : 1), y * len / 2, z / h * r * (fz || 1) * prof * (h < 1e-4 ? 0 : 1));
      const c = white.clone().lerp(tendon, Math.min(1, Math.max(0, (Math.abs(y) - .7) / .3))); col.push(c.r, c.g, c.b);
    }
    g.setAttribute("color", new THREE.Float32BufferAttribute(col, 3)); g.computeVertexNormals();
    const m = new THREE.Mesh(g, mMat); m.position.copy(a).add(b).multiplyScalar(.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize()); muscles.add(m); return m;
  }
  [-1, 1].forEach(s => {
    muscle([s*.02,1.42,.1], [s*.3,1.41,.065], .034, 1, .8); muscle([s*.03,1.32,.105], [s*.3,1.39,.06], .03, 1, .75);   // pectoralis major
    muscle([s*.2,1.52,.01], [s*.34,1.36,.0], .062, 1, 1);                                                             // deltoid
    muscle([s*.32,1.4,.03], [s*.36,1.19,.03], .033);  muscle([s*.32,1.4,-.03], [s*.36,1.19,-.03], .03);               // biceps, triceps
    muscle([s*.36,1.19,.02], [s*.4,.93,.02], .03);    muscle([s*.37,1.19,-.025], [s*.41,.93,-.025], .026);            // forearm flexors, extensors
    muscle([s*.015,1.58,-.06], [s*.2,1.49,-.055], .03, 1, .5); muscle([s*.01,1.55,-.075], [s*.02,1.28,-.09], .045, 1, .4);   // trapezius
    muscle([s*.04,1.02,-.1], [s*.27,1.38,-.06], .05, 1, .4);                                                           // latissimus
    [0,1,2].forEach(i => muscle([s*.04,1.19 - i*.075,.1], [s*.04,1.13 - i*.075,.1], .033, 1, .7));                     // rectus abdominis
    muscle([s*.13,.98,.085], [s*.19,1.2,.07], .035, 1, .6);                                                           // external oblique
    muscle([s*.015,1.52,.045], [s*.06,1.69,-.02], .014);                                                              // sternocleidomastoid
    muscle([s*.1,.99,-.1], [s*.12,.86,-.09], .075, 1, .8);                                                            // gluteus
    muscle([s*.1,.93,.04], [s*.1,.52,.045], .062);    muscle([s*.1,.92,-.045], [s*.1,.52,-.045], .052);               // quadriceps, hamstrings
    muscle([s*.085,.49,-.035], [s*.09,.27,-.03], .034); muscle([s*.115,.49,-.035], [s*.11,.27,-.03], .03);            // gastrocnemius
    muscle([s*.1,.47,.04], [s*.1,.1,.035], .022);                                                                      // tibialis anterior
  });


  /* ---------- real anatomical model (optional .glb) ----------
     Put a real anatomy model at GLB_PATH. If the file is missing, the built-in model above is shown.
     GLB_FIT places the model: y0 = height of its lowest point, h = its total height (in this scene, the body is 1.8 tall).
     Torso organ set: {y0:.82, h:.8}.  Full-body organ set: {y0:0, h:1.8}.  rotY turns it if it faces backwards (Math.PI). */
  const GLB_PATH = "/twin-model.glb";
  const GLB_FIT = {y0:.82, h:.8, rotY:0};
  const real = new THREE.Group(); body.add(real); real.visible = false;
  const realHeart = [], realLungs = [];
  let view = "organs", hasGLB = false;
  function applyView() {
    organs.visible = view === "organs" && !hasGLB; real.visible = view === "organs" && hasGLB;
    muscles.visible = view === "muscle"; skinMat.uniforms.k.value = view === "skin" ? .38 : .1;
  }
  if (THREE.GLTFLoader) new THREE.GLTFLoader().load(GLB_PATH, g => {
    const m = g.scene; m.rotation.y = GLB_FIT.rotY; m.updateMatrixWorld(true);
    let box = new THREE.Box3().setFromObject(m);
    m.scale.multiplyScalar(GLB_FIT.h / box.getSize(new THREE.Vector3()).y); m.updateMatrixWorld(true);
    box = new THREE.Box3().setFromObject(m); const ctr = box.getCenter(new THREE.Vector3());
    m.position.set(m.position.x - ctr.x, m.position.y - box.min.y + GLB_FIT.y0, m.position.z - ctr.z);
    real.add(m); m.updateMatrixWorld(true);
    const hits = [];
    m.traverse(o => {
      if (!o.isMesh) return;
      const mt = o.material; if (mt) { mt.side = THREE.DoubleSide; if (mt.metalness > .15) mt.metalness = .1; }
      const n = (o.name + " " + (mt && mt.name || "")).toLowerCase();
      if (/heart|cardi/.test(n)) hits.push([o, realHeart]); else if (/lung|pulmon/.test(n)) hits.push([o, realLungs]);
    });
    hits.forEach(h => {                       // pivot at the organ's own centre so it beats / breathes in place
      const c = new THREE.Box3().setFromObject(h[0]).getCenter(new THREE.Vector3()), pv = new THREE.Group();
      pv.position.copy(c); real.add(pv); pv.attach(h[0]); h[1].push(pv);
    });
    hasGLB = true; applyView();
  }, undefined, () => {});

  /* ---------- camera + input ---------- */
  let yaw = .35, pitch = .08, dist = 4.4, idle = 0;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const ptr = new Map(); let pinch = 0;
  stage.addEventListener("pointerdown", e => { stage.setPointerCapture(e.pointerId); ptr.set(e.pointerId, [e.clientX, e.clientY]); idle = performance.now(); });
  stage.addEventListener("pointerup", e => ptr.delete(e.pointerId));
  stage.addEventListener("pointercancel", e => ptr.delete(e.pointerId));
  stage.addEventListener("pointermove", e => {
    if (!ptr.has(e.pointerId)) return;
    const p = ptr.get(e.pointerId); idle = performance.now();
    if (ptr.size === 1) { yaw -= (e.clientX - p[0]) * .008; pitch = Math.max(-.5, Math.min(.7, pitch + (e.clientY - p[1]) * .005)); }
    ptr.set(e.pointerId, [e.clientX, e.clientY]);
    if (ptr.size === 2) { const [a, b] = [...ptr.values()], d = Math.hypot(a[0]-b[0], a[1]-b[1]); if (pinch) dist = Math.max(1.6, Math.min(7, dist * pinch / d)); pinch = d; } else pinch = 0;
  });
  stage.addEventListener("wheel", e => { e.preventDefault(); dist = Math.max(1.6, Math.min(7, dist * (1 + e.deltaY * .001))); }, {passive:false});

  function resize() { const w = stage.clientWidth, h = stage.clientHeight; R.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); }
  new ResizeObserver(resize).observe(stage); resize();

  const t0 = performance.now();
  function tick(now) {
    const t = (now - t0) / 1000;
    const dt = Math.min(.05, t - (tick.last || t)); tick.last = t;
    skinMat.uniforms.uT.value = t;
    const gl = (t * 2) % 11 < .12 && Math.floor(t * 2) % 11 === 0;       // brief hologram glitch
    body.position.x = gl ? (Math.random() - .5) * .035 : 0;
    rings.forEach((r, i) => { const f = (t * .1 + i / 3) % 1; r.position.y = .05 + f * 1.9; r.material.opacity = .45 * Math.sin(Math.PI * f); });
    for (let i = 0; i < N; i++) { pa[i*3+1] += pv[i] * dt; if (pa[i*3+1] > 2) pa[i*3+1] = 0; }
    pg.attributes.position.needsUpdate = true;
    cur.hr += (tgt.hr - cur.hr) * .04; cur.rr += (tgt.rr - cur.rr) * .04;
    const ph = (t * cur.hr / 60) % 1;
    const beat = Math.exp(-Math.pow((ph - .08) / .05, 2)) + .6 * Math.exp(-Math.pow((ph - .3) / .06, 2));
    heart.scale.setScalar(1 + .1 * beat);
    pulseU.uPh.value = ph;
    hm.emissiveIntensity = .4 + .5 * beat;
    const br = Math.sin(t * Math.PI * 2 * cur.rr / 60);
    lungs.scale.set(1 + .05 * br, 1 + .035 * br, 1 + .05 * br);
    realHeart.forEach(p => p.scale.setScalar(1 + .07 * beat));
    realLungs.forEach(p => p.scale.set(1 + .05 * br, 1 + .035 * br, 1 + .05 * br));
    body.position.y = .004 * br;
    if (!reduce && now - idle > 4000 && !ptr.size) yaw += .0025;
    cam.position.set(Math.sin(yaw) * Math.cos(pitch) * dist, .95 + Math.sin(pitch) * dist, Math.cos(yaw) * Math.cos(pitch) * dist);
    cam.lookAt(0, .98, 0);
    R.render(scene, cam);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);

  /* ---------- layers ---------- */
  $("#view").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll("#view button").forEach(x => x.setAttribute("aria-pressed", x === b));
    view = b.dataset.v; applyView();
  });

  /* ---------- data ---------- */
  const rows = $("#rows");
  Object.keys(META).forEach(k => {
    const r = document.createElement("div"); r.className = "row"; r.id = "r_" + k;
    r.innerHTML = '<b>' + META[k][0] + '</b><canvas width="168" height="76"></canvas><div class="val">–<small>' + META[k][1] + '</small></div><div class="meta">No reading</div>';
    rows.appendChild(r);
  });

  function ago(iso) {
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return "just now"; if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 129600) return Math.round(s / 3600) + " h ago"; return Math.round(s / 86400) + " days ago";
  }
  const SRC = {ble:"Bluetooth", esp32:"ESP32", manual:"Typed in"};

  function spark(cv, pts) {
    const g = cv.getContext("2d"), w = cv.width, h = cv.height; g.clearRect(0, 0, w, h);
    if (pts.length < 2) return;
    const lo = Math.min(...pts), hi = Math.max(...pts), sp = hi - lo || 1;
    g.beginPath(); pts.forEach((p, i) => { const x = i / (pts.length - 1) * (w - 6) + 3, y = h - 6 - (p - lo) / sp * (h - 12); i ? g.lineTo(x, y) : g.moveTo(x, y); });
    g.strokeStyle = "#5ff3ff"; g.lineWidth = 2.5; g.lineJoin = "round"; g.stroke();
  }

  function render() {
    let any = false;
    Object.keys(META).forEach(k => {
      const L = state.latest[k], r = $("#r_" + k);
      if (L) any = true;
      r.querySelector(".val").innerHTML = (L ? (k === "bp" ? Math.round(L.value) + "/" + Math.round(L.value2) : L.value) : "–") + "<small>" + META[k][1] + "</small>";
      r.querySelector(".meta").textContent = L ? (SRC[L.source] || L.source) + " · " + ago(L.measured_at) : "No reading";
      spark(r.querySelector("canvas"), (state.history[k] || []).map(h => h.value));
    });
    $("#empty").style.display = any ? "none" : "block";
    if (state.latest.heart_rate) tgt.hr = Math.max(30, Math.min(200, state.latest.heart_rate.value));
    if (state.latest.resp_rate) tgt.rr = Math.max(6, Math.min(40, state.latest.resp_rate.value));
  }

  function setLive(on) { $("#dot").classList.toggle("on", on); $("#live").textContent = on ? "Live" : "Reconnecting…"; }
  async function load() {
    try { const r = await fetch("/api/twin/state?days=" + days, {credentials:"same-origin"}); if (!r.ok) throw 0; state = await r.json(); render(); }
    catch (e) { setLive(false); }
  }
  $("#range").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    document.querySelectorAll("#range button").forEach(x => x.setAttribute("aria-pressed", x === b));
    days = +b.dataset.d; load();
  });

  load(); setInterval(load, 45000);
  if (window.EventSource) {
    const es = new EventSource("/api/twin/stream");
    es.addEventListener("twin_ready", () => setLive(true));
    es.addEventListener("twin_update", load);
    es.onerror = () => setLive(false);
  }
})();
</script>
</body>
</html>
'''


@app.route("/twin")
@login_required
def twin():
    page = TWIN_HTML.replace("{{ patient_id }}", "MB-PT-%06d" % g.user["id"])
    resp = app.response_class(page, mimetype="text/html")
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/twin-model.glb")
@login_required
def twin_model():
    """Optional real anatomy model: put the file at static/models/twin_organs.glb. Without it the built-in model is used."""
    try:
        resp = app.send_static_file("models/twin_organs.glb")
    except Exception:
        abort(404)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


@app.route("/report")
@login_required
def report():
    return render_template("report.html")


@app.route("/reminders")
@login_required
def reminders():
    return render_template("reminders.html")


@app.route("/assistant")
@login_required
def assistant():
    return render_template("assistant.html")


# ---------- Simple per-user rate limit (protects your API bill) ----------
import time
_RATE = {}
_IP_RATE = _RATE                                  # old name kept: one shared table for user and address limits


def _hit(bucket, key, limit, window):
    """Sliding-window limiter. True if this call is allowed (and counted)."""
    now = time.time()
    k = (bucket, str(key))
    hits = [t for t in _RATE.get(k, ()) if now - t < window]
    allowed = len(hits) < limit
    if allowed:
        hits.append(now)
    _RATE[k] = hits
    if len(_RATE) > 4000:                         # forget idle keys only. Never clear active ones, or an attacker could reset every limit
        for stale in [s for s, v in _RATE.items() if not v or now - v[-1] > 3600]:
            del _RATE[stale]
    return allowed


def rate_ok(bucket, limit, window=3600):
    return _hit(bucket, g.user["id"], limit, window)


# ---------- JSON APIs ----------
@app.route("/api/report", methods=["POST"])
@login_required
def api_report():
    f = request.files.get("report")
    if not f or not f.filename:
        return jsonify(error="Choose a file to upload."), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify(error="Unsupported file type. Use PDF, JPG, PNG or TXT."), 400
    if request.form.get("ack") != "1":
        return jsonify(error="Please confirm you removed personal details from the report."), 400
    if ai_configured():
        blocked = consent_blocked("ai_processing", "AI processing")
        if blocked:
            return blocked
    if ai_configured() and not rate_ok("report", 10):
        return jsonify(error="Report limit reached (10 per hour). Please try again later."), 429
    lang = request.form.get("lang", "English")
    lang = lang if lang in ("English", "Tamil", "Hindi") else "English"
    data = f.read()  # processed in memory; reports are NOT saved to disk
    try:
        result = explain_report(os.path.basename(f.filename), data, lang)
        audit("ai_report")
        return jsonify(result)
    except AIError as e:
        app.logger.warning("AI report error: %s", str(e)[:200])
        return jsonify(error=str(e)), 502


@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    body = request.get_json(silent=True) or {}
    msg = str(body.get("message", "")).strip()
    if not msg:
        return jsonify(error="Type a question first."), 400
    if len(msg) > 1000:
        return jsonify(error="Please keep your question under 1000 characters."), 400
    mode = body.get("mode", "chat")
    mode = mode if mode in ("chat", "term", "appointment") else "chat"
    lang = body.get("lang", "English")
    lang = lang if lang in LANGS else "English"
    if ai_configured() and not is_emergency(msg):
        blocked = consent_blocked("ai_processing", "AI processing")
        if blocked:
            return blocked
    if ai_configured() and not rate_ok("chat", 40):
        return jsonify(error="Message limit reached (40 per hour). Please try again later."), 429
    try:
        result = chat_reply(msg, body.get("history", []), mode, lang)
        if not result.get("emergency"):
            audit("ai_chat", mode)
        return jsonify(result)
    except AIError as e:
        app.logger.warning("AI chat error: %s", str(e)[:200])
        return jsonify(error=str(e)), 502


# ---------- Platform status, consent, audit and data rights ----------
STATUS_LABEL = {"impl": "Implemented", "part": "Partial", "demo": "Demo data", "exp": "Experimental", "plan": "Planned"}

CONSENT_PURPOSES = {
    "ai_processing": {"title": "AI processing",
                      "desc": "Send my questions and uploaded reports to the AI provider so it can write explanations. Live Medical Research also sends the topic I type to a web-search provider (Olostep). MediBridge AI does not save my questions."},
    "push_alerts": {"title": "Reminder alerts",
                    "desc": "Store a push address for my browser or phone so reminder alerts can be sent to it."},
}


def feature_list():
    ai_note = "AI key connected." if ai_configured() else "Needs GEMINI_API_KEY or AI_API_KEY on the server."
    return [
        dict(name="Account, login and sessions", status="impl", note="Hashed passwords, CSRF tokens, secure cookies, log-in rate limit."),
        dict(name="AI Health Companion", status="impl" if ai_configured() else "part", note="Education only, never diagnoses. " + ai_note),
        dict(name="Smart appointment preparation", status="impl" if ai_configured() else "part", note="AI drafts questions to ask. " + ai_note),
        dict(name="Live Medical Research", status="impl" if research_configured() else "part",
             note="Olostep searches public sources, preferring WHO, CDC, NHS and government sites. The AI explains them with links and a last-checked date. Results are cached to save credits. "
                  + ("Connected." if research_configured() else "Needs OLOSTEP_API_KEY and the AI key on the server.")),
        dict(name="Medical Report Explainer", status="part", note="Uploads are explained by the AI provider and are not clinically validated. The sample is fictional."),
        dict(name="Medicine reminders and push alerts", status="part", note="Server scheduler is built. Delivery depends on the phone and must be confirmed on your own device."),
        dict(name="Patient consent management", status="impl", note="Stored per account and enforced for AI processing and alerts."),
        dict(name="Audit log, data export and deletion", status="impl", note="Activity kept 180 days. You can download or delete everything."),
        dict(name="Emergency information", status="impl", note="Your contact is kept only in this browser."),
        dict(name="Digital Health Timeline", status="part", note="Reminders and entries with a phone alert are real and saved to your account. Other entries are demo samples or browser-only notes."),
        dict(name="Timeline phone alerts", status="part", note="Server scheduler sends a push at the time you choose. Needs alerts turned on and the server awake. Delivery must be confirmed on your own device."),
        dict(name="Health Knowledge Hub", status="part", note="Topic shortcuts open AI explanations. Needs the AI key."),
        dict(name="Explainable AI insights", status="part", note="Range checks show the exact rule used. No predictive model exists yet."),
        dict(name="Smart Medical Device Hub", status="part", note="Supports Web Bluetooth LE pairing, ESP32 HTTP API ingestion, and clearly labelled demo mode."),
        dict(name="Medical Device Digital Passport", status="impl", note="Record model, firmware and calibration dates per device. Self-reported and not verified."),
        dict(name="Manual health readings and trend chart", status="impl", note="Type in readings, see charts, statistics and a fitted line. Describes your entries only. Does not diagnose or predict."),
        dict(name="Trend analysis with AI", status="part", note="Statistical trends are live. The AI only helps you prepare doctor questions. AI-written interpretation is not built and needs clinical validation."),
        dict(name="Sample charts (demo)", status="demo", note="The sample charts use simulated data and say so on the chart."),
        dict(name="Tamil voice assistant", status="exp", note="Uses the browser's read-aloud and dictation. Quality depends on the phone's Tamil voices."),
        dict(name="Bluetooth device integration", status="part", note="Web Bluetooth GATT integration built for Heart Rate, SpO2, Thermometer, BP, and ESP32 BLE JSON in Chrome/Edge."),
        dict(name="Digital Patient Twin", status="part", note="Interactive 3D Three.js body with organ views, BLE & ESP32 live stream, trend graphs, and Gemini AI summary. Educational prototype; not clinically validated."),
        dict(name="Secure healthcare data sharing", status="plan", note="Time-limited, consented sharing with a named clinician. Needs access controls and a security review first."),
    ]


def live_status():
    prov = ai_provider()
    return [
        ("AI service", bool(prov), ("connected (%s)" % prov) if prov else "not connected"),
        ("Research search", olostep_configured(), "connected (Olostep)" if olostep_configured() else "not connected"),
        ("Database", True, "PostgreSQL" if USE_PG else "SQLite (local)"),
        ("Push library", PUSH_OK, "ready" if PUSH_OK else "pywebpush missing"),
        ("Reminder scheduler", _SCHED["started"], "running" if _SCHED["started"] else "off"),
    ]


def get_consents():
    return {r["purpose"]: bool(r["granted"]) for r in
            query_all("SELECT purpose, granted FROM consents WHERE user_id = ?", (g.user["id"],))}


def has_consent(purpose):
    row = query("SELECT granted FROM consents WHERE user_id = ? AND purpose = ?", (g.user["id"], purpose), one=True)
    return bool(row and row["granted"])


def consent_blocked(purpose, what):
    """JSON error telling the page to ask for consent. Returns None when consent exists."""
    if has_consent(purpose):
        return None
    return jsonify(error="Please allow %s in the Privacy centre first." % what, code="consent_required"), 403


def _utcnow():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def audit(event, detail="", user_id=None):
    """Record a security-relevant event. Never put message or report content in here."""
    try:
        uid = user_id if user_id is not None else (g.user["id"] if g.get("user") else None)
        query("INSERT INTO audit_log (user_id, event, detail, created_at) VALUES (?, ?, ?, ?)",
              (uid, event[:40], str(detail)[:80], _utcnow()), commit=True)
    except Exception as e:
        app.logger.warning("audit failed: %s", type(e).__name__)
        try:
            get_db().rollback()
        except Exception:
            pass


def purge_audit():
    """Retention policy: audit records older than 180 days are deleted."""
    try:
        with app.app_context():
            cutoff = (datetime.utcnow() - timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S")
            query("DELETE FROM audit_log WHERE created_at < ?", (cutoff,), commit=True)
    except Exception as e:
        print("Audit purge skipped:", type(e).__name__)


def ip_rate_ok(bucket, limit, window):
    """Rate limit by network address (for log-in and sign-up, where nobody is logged in yet)."""
    return _hit(bucket, request.remote_addr or "?", limit, window)


@app.route("/api/consents", methods=["GET"])
@login_required
def api_consents_list():
    return jsonify(consents=get_consents())


@app.route("/api/consents", methods=["POST"])
@login_required
def api_consents_set():
    body = request.get_json(silent=True) or {}
    purpose, granted = body.get("purpose"), bool(body.get("granted"))
    if purpose not in CONSENT_PURPOSES:
        return jsonify(error="Unknown consent type."), 400
    if not rate_ok("consent", 60):
        return jsonify(error="Too many changes. Please try again later."), 429
    uid = g.user["id"]
    query("DELETE FROM consents WHERE user_id = ? AND purpose = ?", (uid, purpose))
    query("INSERT INTO consents (user_id, purpose, granted, updated_at) VALUES (?, ?, ?, ?)",
          (uid, purpose, 1 if granted else 0, _utcnow()), commit=True)
    if purpose == "push_alerts" and not granted:          # withdrawing consent removes the stored push addresses
        query("DELETE FROM push_subs WHERE user_id = ?", (uid,), commit=True)
    audit("consent_granted" if granted else "consent_withdrawn", purpose)
    return jsonify(consents=get_consents())


@app.route("/api/audit")
@login_required
def api_audit():
    rows = query_all("SELECT event, detail, created_at FROM audit_log WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                     (g.user["id"],))
    return jsonify(events=[{"event": r["event"], "detail": r["detail"], "created_at": str(r["created_at"])} for r in rows])


@app.route("/api/privacy/export")
@login_required
def api_privacy_export():
    uid = g.user["id"]
    acct = query("SELECT name, email, created_at FROM users WHERE id = ?", (uid,), one=True)
    data = {
        "exported_at_utc": _utcnow(),
        "account": {"name": acct["name"], "email": acct["email"], "created_at": str(acct["created_at"])},
        "reminders": [dict(r, created_at=str(r["created_at"])) for r in query_all(
            "SELECT name, remind_time, note, created_at FROM reminders WHERE user_id = ? ORDER BY id", (uid,))],
        "consents": [dict(r) for r in query_all("SELECT purpose, granted, updated_at FROM consents WHERE user_id = ?", (uid,))],
        "device_passports": [dict(r) for r in query_all(
            "SELECT label, dtype, maker, model, serial_tail, firmware, first_use, last_cal, next_cal, reg_status, notes "
            "FROM device_passports WHERE user_id = ? ORDER BY id", (uid,))],
        "twin_esp32_tokens": [dict(r) for r in query_all(
            "SELECT label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id", (uid,))],
        "health_readings": [dict(r) for r in query_all(
            "SELECT metric, value, value2, measured_at AS measured_at_utc, context, note, source FROM health_readings "
            "WHERE user_id = ? ORDER BY measured_at", (uid,))],
        "reading_ranges": [dict(r) for r in query_all("SELECT metric, low, high FROM reading_ranges WHERE user_id = ?", (uid,))],
        "timeline_alerts": [dict(r) for r in query_all(
            "SELECT etype AS type, title, when_utc, notify_min FROM timeline_events WHERE user_id = ? ORDER BY when_utc", (uid,))],
        "alert_devices_registered": len(query_all("SELECT id FROM push_subs WHERE user_id = ?", (uid,))),
        "activity_log": [dict(r, created_at=str(r["created_at"])) for r in query_all(
            "SELECT event, detail, created_at FROM audit_log WHERE user_id = ? ORDER BY id", (uid,))],
        "not_stored": "Chat messages and uploaded reports are not stored by MediBridge AI.",
    }
    audit("data_export")
    resp = jsonify(data)
    resp.headers["Content-Disposition"] = "attachment; filename=medibridge-my-data.json"
    return resp


@app.route("/api/privacy/delete", methods=["POST"])
@login_required
def api_privacy_delete():
    body = request.get_json(silent=True) or {}
    if not rate_ok("delete", 5):
        return jsonify(error="Too many attempts. Please try again later."), 429
    row = query("SELECT password_hash FROM users WHERE id = ?", (g.user["id"],), one=True)
    if not row or not check_password_hash(row["password_hash"], str(body.get("password", ""))):
        audit("delete_failed", "wrong password")
        return jsonify(error="Incorrect password."), 403
    uid = g.user["id"]
    for sql in ("DELETE FROM reminder_log WHERE reminder_id IN (SELECT id FROM reminders WHERE user_id = ?)",
                "DELETE FROM reminders WHERE user_id = ?", "DELETE FROM push_subs WHERE user_id = ?",
                "DELETE FROM consents WHERE user_id = ?",
                "DELETE FROM health_readings WHERE user_id = ?", "DELETE FROM reading_ranges WHERE user_id = ?",
                "DELETE FROM device_passports WHERE user_id = ?",
                "DELETE FROM twin_tokens WHERE user_id = ?",
                "DELETE FROM event_alerts_sent WHERE event_id IN (SELECT id FROM timeline_events WHERE user_id = ?)",
                "DELETE FROM timeline_events WHERE user_id = ?",
                "DELETE FROM audit_log WHERE user_id = ?"):
        query(sql, (uid,))
    query("DELETE FROM users WHERE id = ?", (uid,), commit=True)      # one transaction for everything
    session.clear()
    app.logger.info("account %s deleted by owner", uid)
    return jsonify(ok=True)


# ---------- Medical Device Digital Passport (self-reported device records) ----------
DEVICE_TYPES = {"pulse_oximeter": "Pulse oximeter", "heart_rate": "Heart-rate sensor", "thermometer": "Thermometer",
                "bp_monitor": "Blood pressure monitor", "glucose_meter": "Glucose meter",
                "respiratory": "Respiratory sensor", "other": "Other device"}
REG_STATUS = {"unknown": "Not known", "approved": "Approved or cleared in my country", "ce": "CE marked",
              "fda": "FDA cleared", "wellness": "Wellness device, not medical"}
MAX_DEVICES = 10


def _clean_date(v):
    """'' for empty, 'YYYY-MM-DD' for a valid date, None for anything else."""
    v = str(v or "").strip()
    if not v:
        return ""
    try:
        return datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


def _cal_status(next_cal):
    if not next_cal:
        return "not_set"
    days = (datetime.strptime(next_cal, "%Y-%m-%d").date() - datetime.utcnow().date()).days
    return "overdue" if days < 0 else ("due_soon" if days <= 30 else "ok")


def _user_devices():
    rows = query_all(
        "SELECT id, label, dtype, maker, model, serial_tail, firmware, first_use, last_cal, next_cal, reg_status, notes "
        "FROM device_passports WHERE user_id = ? ORDER BY id DESC", (g.user["id"],))
    for r in rows:
        r["cal_status"] = _cal_status(r["next_cal"])
    return rows


@app.route("/api/devices", methods=["GET"])
@login_required
def api_devices_list():
    return jsonify(devices=_user_devices())


@app.route("/api/devices", methods=["POST"])
@login_required
def api_devices_add():
    b = request.get_json(silent=True) or {}
    label = str(b.get("label", "")).strip()[:60]
    if not label:
        return jsonify(error="Give the device a name."), 400
    dtype, reg = b.get("dtype", "other"), b.get("reg_status", "unknown")
    if dtype not in DEVICE_TYPES or reg not in REG_STATUS:
        return jsonify(error="Choose a valid device type and approval status."), 400
    dates = [_clean_date(b.get(k)) for k in ("first_use", "last_cal", "next_cal")]
    if None in dates:
        return jsonify(error="Dates must look like 2026-11-21."), 400
    if not rate_ok("device", 40):
        return jsonify(error="Too many changes. Please try again later."), 429
    if len(_user_devices()) >= MAX_DEVICES:
        return jsonify(error="You can save up to %d device passports. Delete one first." % MAX_DEVICES), 400
    serial_tail = re.sub(r"[^A-Za-z0-9]", "", str(b.get("serial_tail", "")))[-4:]      # last 4 only (data minimisation)
    query("INSERT INTO device_passports (user_id, label, dtype, maker, model, serial_tail, firmware, first_use, "
          "last_cal, next_cal, reg_status, notes, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
          (g.user["id"], label, dtype, str(b.get("maker", "")).strip()[:60], str(b.get("model", "")).strip()[:60],
           serial_tail, str(b.get("firmware", "")).strip()[:30], dates[0], dates[1], dates[2], reg,
           str(b.get("notes", "")).strip()[:200], _utcnow()), commit=True)
    audit("device_added", dtype)
    return jsonify(devices=_user_devices())


@app.route("/api/devices/<int:did>/delete", methods=["POST"])
@login_required
def api_devices_delete(did):
    query("UPDATE health_readings SET device_id = NULL WHERE device_id = ? AND user_id = ?", (did, g.user["id"]))
    query("DELETE FROM device_passports WHERE id = ? AND user_id = ?", (did, g.user["id"]), commit=True)
    audit("device_deleted")
    return jsonify(devices=_user_devices())


# ---------- Manual health readings and statistical trends ----------
# These routes only store and describe numbers the user typed in. They never label a value as good or bad,
# diagnose, or predict. "Plausible range" checks below only catch typing mistakes.
READING_METRICS = {
    "bp": {"label": "Blood pressure", "unit": "mmHg", "two": True, "lo": 50, "hi": 260, "lo2": 30, "hi2": 160},
    "heart_rate": {"label": "Heart rate", "unit": "bpm", "two": False, "lo": 20, "hi": 250},
    "spo2": {"label": "Oxygen saturation (SpO2)", "unit": "%", "two": False, "lo": 50, "hi": 100},
    "temperature": {"label": "Body temperature", "unit": "°C", "two": False, "lo": 30, "hi": 45},
    "glucose": {"label": "Blood glucose", "unit": "mg/dL", "two": False, "lo": 20, "hi": 800},
    "weight": {"label": "Weight", "unit": "kg", "two": False, "lo": 2, "hi": 400},
    "resp_rate": {"label": "Respiratory rate", "unit": "breaths/min", "two": False, "lo": 4, "hi": 60},
}
RANGE_KEYS = {"bp_sys", "bp_dia", "heart_rate", "spo2", "temperature", "glucose", "weight", "resp_rate"}
READING_CONTEXTS = {"", "fasting", "after_meal", "resting", "after_activity"}
MAX_READINGS = 2000
# two-sided 95% critical t values by degrees of freedom (largest listed df not above the real df is used)
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}
_TS = "%Y-%m-%d %H:%M:%S"


def _num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _parse_when(s):
    """ISO time from the browser -> naive UTC datetime, or None if invalid / unreasonable."""
    try:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is not None:
        d = (d - d.utcoffset()).replace(tzinfo=None)
    now = datetime.utcnow()
    if d > now + timedelta(minutes=10) or d < now - timedelta(days=3650):
        return None
    return d.replace(microsecond=0)


def _iso(ts):
    return datetime.strptime(ts, _TS).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_ranges():
    return {r["metric"]: {"low": r["low"], "high": r["high"]}
            for r in query_all("SELECT metric, low, high FROM reading_ranges WHERE user_id = ?", (g.user["id"],))}


def _trend_series(rows, key, label, unit, field):
    """Plain descriptive statistics plus a least-squares line. No prediction."""
    pts = [(datetime.strptime(r["measured_at"], _TS), float(r[field])) for r in rows if r[field] is not None]
    out = {"key": key, "label": label, "unit": unit, "stats": None, "fit": None, "note": ""}
    if not pts:
        out["note"] = "No readings in this period."
        return out
    vals, n, t0 = [v for _, v in pts], len(pts), pts[0][0]
    xs = [(t - t0).total_seconds() / 86400 for t, _ in pts]
    mean = sum(vals) / n
    st = {"n": n, "latest": vals[-1], "latest_at": pts[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ"),
          "min": min(vals), "max": max(vals), "mean": round(mean, 1)}
    span = xs[-1]
    if n >= 5 and span >= 1:
        mx = sum(xs) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (v - mean) for x, v in zip(xs, vals)) / sxx
        intercept = mean - slope * mx
        change = slope * span
        sse = sum((v - (intercept + slope * x)) ** 2 for x, v in zip(xs, vals))
        if sse < 1e-9:                                  # points lie exactly on a straight line
            direction = "no clear direction" if abs(slope) < 1e-9 else ("rising" if slope > 0 else "falling")
        else:                                           # a direction is named only if the slope clearly beats the scatter (t-test, 95%)
            se = math.sqrt(sse / (n - 2) / sxx)
            crit = 1.96
            for k in sorted(_T95):
                if k <= n - 2:
                    crit = _T95[k]
            direction = "no clear direction" if abs(slope / se) < crit else ("rising" if slope > 0 else "falling")
        st.update(direction=direction, slope_per_week=round(slope * 7, 2), fitted_change=round(change, 1))
        out["fit"] = {"t0": pts[0][0].strftime("%Y-%m-%dT%H:%M:%SZ"), "v0": round(intercept, 2),
                      "t1": pts[-1][0].strftime("%Y-%m-%dT%H:%M:%SZ"), "v1": round(intercept + slope * span, 2)}
    else:
        out["note"] = "Add at least 5 readings over 2 or more days to see a trend line."
    out["stats"] = st
    return out


def _reading_rows(metric="", days=None, limit=MAX_READINGS):
    sql = ("SELECT r.id, r.metric, r.value, r.value2, r.measured_at, r.context, r.note, r.device_id, d.label AS device_label "
           "FROM health_readings r LEFT JOIN device_passports d ON d.id = r.device_id AND d.user_id = r.user_id "
           "WHERE r.user_id = ?")
    params = [g.user["id"]]
    if metric:
        sql += " AND r.metric = ?"
        params.append(metric)
    if days:
        sql += " AND r.measured_at >= ?"
        params.append((datetime.utcnow() - timedelta(days=days)).strftime(_TS))
    sql += " ORDER BY r.measured_at ASC, r.id ASC LIMIT %d" % int(limit)
    return query_all(sql, tuple(params))


@app.route("/api/readings", methods=["GET"])
@login_required
def api_readings_list():
    metric = request.args.get("metric", "")
    if metric and metric not in READING_METRICS:
        return jsonify(error="Unknown measurement."), 400
    raw = request.args.get("days", "30")
    days = None
    if raw != "all":
        try:
            days = int(raw)
        except ValueError:
            return jsonify(error="Invalid period."), 400
        if not 1 <= days <= 3650:
            return jsonify(error="Invalid period."), 400
    rows = _reading_rows(metric, days)
    readings = [dict(r, measured_at=_iso(r["measured_at"])) for r in rows]
    resp = {"readings": readings, "series": [], "ranges": _get_ranges()}
    if metric:
        m = READING_METRICS[metric]
        resp.update(metric=metric, label=m["label"], unit=m["unit"])
        if m["two"]:
            resp["series"] = [_trend_series(rows, "bp_sys", "Systolic", m["unit"], "value"),
                              _trend_series(rows, "bp_dia", "Diastolic", m["unit"], "value2")]
        else:
            resp["series"] = [_trend_series(rows, metric, m["label"], m["unit"], "value")]
    return jsonify(resp)


@app.route("/api/readings", methods=["POST"])
@login_required
def api_readings_add():
    b = request.get_json(silent=True) or {}
    metric = b.get("metric")
    if metric not in READING_METRICS:
        return jsonify(error="Choose a measurement type."), 400
    m = READING_METRICS[metric]
    v = _num(b.get("value"))
    if v is None:
        return jsonify(error="Enter the reading as a number."), 400
    unit = str(b.get("unit", ""))
    if metric == "temperature" and unit == "F":
        v = (v - 32) * 5 / 9
    elif metric == "glucose" and unit == "mmol/L":
        v = v * 18.016
    v = round(v, 1)
    v2 = None
    if m["two"]:
        v2 = _num(b.get("value2"))
        if v2 is None:
            return jsonify(error="Enter both blood pressure numbers."), 400
        v2 = round(v2, 1)
        if not m["lo2"] <= v2 <= m["hi2"] or v <= v2:
            return jsonify(error="Please check the blood pressure numbers. The top number should be larger than the bottom one."), 400
    if not m["lo"] <= v <= m["hi"]:
        return jsonify(error="That value looks like a typing mistake for %s. Please check it." % m["label"].lower()), 400
    when = _parse_when(b.get("measured_at"))
    if when is None:
        return jsonify(error="Choose a valid date and time (not in the future)."), 400
    ctx = str(b.get("context", "") or "")
    if ctx not in READING_CONTEXTS:
        return jsonify(error="Unknown situation."), 400
    dev = b.get("device_id")
    dev_id = None
    if dev not in (None, ""):
        try:
            dev_id = int(dev)
        except (TypeError, ValueError):
            return jsonify(error="Unknown device."), 400
        if not query("SELECT 1 FROM device_passports WHERE id = ? AND user_id = ?", (dev_id, g.user["id"]), one=True):
            return jsonify(error="Unknown device."), 400
    if not rate_ok("reading", 120):
        return jsonify(error="Too many readings this hour. Please try again later."), 429
    if query("SELECT COUNT(*) AS n FROM health_readings WHERE user_id = ?", (g.user["id"],), one=True)["n"] >= MAX_READINGS:
        return jsonify(error="You reached the limit of %d readings. Download your data and delete old readings." % MAX_READINGS), 400
    query("INSERT INTO health_readings (user_id, metric, value, value2, measured_at, context, note, device_id, source, created_at) "
          "VALUES (?,?,?,?,?,?,?,?,?,?)",
          (g.user["id"], metric, v, v2, when.strftime(_TS), ctx, str(b.get("note", "")).strip()[:120], dev_id, "manual", _utcnow()),
          commit=True)
    _twin_notify(g.user["id"])
    audit("reading_added", metric)
    return jsonify(ok=True)


@app.route("/api/readings/<int:rid>/delete", methods=["POST"])
@login_required
def api_readings_delete(rid):
    query("DELETE FROM health_readings WHERE id = ? AND user_id = ?", (rid, g.user["id"]), commit=True)
    audit("reading_deleted")
    return jsonify(ok=True)


@app.route("/api/ranges", methods=["GET"])
@login_required
def api_ranges_get():
    return jsonify(ranges=_get_ranges())


@app.route("/api/ranges", methods=["POST"])
@login_required
def api_ranges_set():
    """A reference range the user copied from their own doctor or report. Used only to draw a band and compare."""
    b = request.get_json(silent=True) or {}
    key = b.get("metric")
    if key not in RANGE_KEYS:
        return jsonify(error="Unknown measurement."), 400
    low = _num(b.get("low")) if str(b.get("low", "")).strip() != "" else None
    high = _num(b.get("high")) if str(b.get("high", "")).strip() != "" else None
    if (str(b.get("low", "")).strip() != "" and low is None) or (str(b.get("high", "")).strip() != "" and high is None):
        return jsonify(error="Enter the range as numbers."), 400
    if low is not None and high is not None and low > high:
        return jsonify(error="The lower number must not be larger than the upper number."), 400
    if not rate_ok("range", 60):
        return jsonify(error="Too many changes. Please try again later."), 429
    query("DELETE FROM reading_ranges WHERE user_id = ? AND metric = ?", (g.user["id"], key))
    if low is None and high is None:
        get_db().commit()
    else:
        query("INSERT INTO reading_ranges (user_id, metric, low, high) VALUES (?,?,?,?)", (g.user["id"], key, low, high), commit=True)
    audit("range_saved", key)
    return jsonify(ranges=_get_ranges())


def _csv_safe(s):
    s = str(s or "")
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


@app.route("/api/readings/export.csv")
@login_required
def api_readings_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["measured_at_utc", "measurement", "value", "value_2", "unit", "situation", "device", "note", "source"])
    for r in _reading_rows():
        m = READING_METRICS.get(r["metric"], {"label": r["metric"], "unit": ""})
        w.writerow([r["measured_at"], m["label"], r["value"], "" if r["value2"] is None else r["value2"], m["unit"],
                    _csv_safe(r["context"]), _csv_safe(r["device_label"]), _csv_safe(r["note"]), "manual entry"])
    audit("readings_export")
    return app.response_class(buf.getvalue(), mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=medibridge-readings.csv"})


# ---------- Digital Patient Twin: Real-Time SSE, BLE Save, ESP32 Ingestion & AI Analysis ----------
TWIN_METRICS = ("heart_rate", "spo2", "temperature", "resp_rate", "bp")
_TWIN_SUBS = {}
_TWIN_LOCK = threading.Lock()


def _twin_notify(uid):
    with _TWIN_LOCK:
        queues = list(_TWIN_SUBS.get(uid, ()))
    for q in queues:
        try:
            q.put_nowait({"ts": _utcnow()})
        except queue.Full:
            pass


def _save_sensor_reading(uid, metric, val, val2=None, measured_at=None, source="ble", note=""):
    if metric not in TWIN_METRICS:
        return False, "Unsupported Twin metric."
    m = READING_METRICS[metric]
    v = _num(val)
    if v is None:
        return False, "Value must be numeric."
    v = round(v, 1)
    v2 = None
    if m["two"]:
        v2 = _num(val2)
        if v2 is None:
            return False, "Both systolic and diastolic values are required."
        v2 = round(v2, 1)
        if not m["lo2"] <= v2 <= m["hi2"] or v <= v2:
            return False, "Invalid blood pressure numbers."
    if not m["lo"] <= v <= m["hi"]:
        return False, "Value out of plausible range for %s." % m["label"].lower()
    when = _parse_when(measured_at) if measured_at else datetime.utcnow().replace(microsecond=0)
    if when is None:
        when = datetime.utcnow().replace(microsecond=0)
    if query("SELECT COUNT(*) AS n FROM health_readings WHERE user_id = ?", (uid,), one=True)["n"] >= MAX_READINGS:
        oldest = query("SELECT id FROM health_readings WHERE user_id = ? ORDER BY measured_at ASC, id ASC LIMIT 1", (uid,), one=True)
        if oldest:
            query("DELETE FROM health_readings WHERE id = ?", (oldest["id"],))
    query("INSERT INTO health_readings (user_id, metric, value, value2, measured_at, context, note, device_id, source, created_at) "
          "VALUES (?,?,?,?,?,?,?,?,?,?)",
          (uid, metric, v, v2, when.strftime(_TS), "", str(note)[:120], None, source[:16], _utcnow()), commit=True)
    return True, None


def _twin_snapshot(uid, days=1):
    latest = {}
    history = {}
    cutoff = (datetime.utcnow() - timedelta(days=days)).strftime(_TS)
    for k in TWIN_METRICS:
        last_row = query(
            "SELECT value, value2, measured_at, source FROM health_readings "
            "WHERE user_id = ? AND metric = ? ORDER BY measured_at DESC, id DESC LIMIT 1",
            (uid, k), one=True)
        if last_row:
            latest[k] = {"value": last_row["value"], "value2": last_row["value2"],
                         "measured_at": _iso(last_row["measured_at"]), "source": last_row["source"] or "manual"}
        rows = query_all(
            "SELECT value, value2, measured_at, source FROM health_readings "
            "WHERE user_id = ? AND metric = ? AND measured_at >= ? ORDER BY measured_at ASC, id ASC LIMIT 200",
            (uid, k, cutoff))
        history[k] = [{"value": r["value"], "value2": r["value2"], "measured_at": _iso(r["measured_at"]),
                       "source": r["source"] or "manual"} for r in rows]
    recent_rows = query_all(
        "SELECT metric, value, value2, measured_at, source FROM health_readings "
        "WHERE user_id = ? AND metric IN ('heart_rate','spo2','temperature','resp_rate','bp') "
        "ORDER BY measured_at DESC, id DESC LIMIT 20", (uid,))
    recent = [{"metric": r["metric"], "value": r["value"], "value2": r["value2"],
               "measured_at": _iso(r["measured_at"]), "source": r["source"] or "manual"} for r in recent_rows]
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (uid,))
    return {"latest": latest, "history": history, "recent": recent, "tokens": tokens}


@app.route("/api/twin/state")
@login_required
def api_twin_state():
    try:
        days = int(request.args.get("days", "1"))
    except ValueError:
        days = 1
    days = days if days in (1, 7, 30) else 1
    return jsonify(_twin_snapshot(g.user["id"], days))


@app.route("/api/twin/ble-reading", methods=["POST"])
@login_required
def api_twin_ble_reading():
    if not rate_ok("twin_ble", 360):
        return jsonify(error="Too many BLE saves this hour."), 429
    b = request.get_json(silent=True) or {}
    ok, err = _save_sensor_reading(g.user["id"], b.get("metric"), b.get("value"), b.get("value2"), b.get("measured_at"), "ble")
    if not ok:
        return jsonify(error=err), 400
    _twin_notify(g.user["id"])
    return jsonify(ok=True)


@app.route("/api/twin/tokens", methods=["POST"])
@login_required
def api_twin_token_create():
    b = request.get_json(silent=True) or {}
    label = str(b.get("label", "")).strip()[:60] or "ESP32 Node"
    uid = g.user["id"]
    if not rate_ok("twin_tok", 20):
        return jsonify(error="Too many token changes. Please try again later."), 429
    if query("SELECT COUNT(*) AS n FROM twin_tokens WHERE user_id = ?", (uid,), one=True)["n"] >= 10:
        return jsonify(error="Limit of 10 ESP32 tokens reached. Revoke an unused one first."), 400
    raw = "mb_esp32_" + secrets.token_urlsafe(24)
    thash = hashlib.sha256(raw.encode()).hexdigest()
    query("INSERT INTO twin_tokens (user_id, label, token_hash, token_prefix, last_seen, created_at) VALUES (?,?,?,?,?,?)",
          (uid, label, thash, raw[:14], "", _utcnow()), commit=True)
    audit("twin_token_created", label)
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (uid,))
    return jsonify(token=raw, tokens=tokens)


@app.route("/api/twin/tokens/<int:tid>/delete", methods=["POST"])
@login_required
def api_twin_token_delete(tid):
    query("DELETE FROM twin_tokens WHERE id = ? AND user_id = ?", (tid, g.user["id"]), commit=True)
    audit("twin_token_revoked")
    tokens = query_all("SELECT id, label, token_prefix, last_seen, created_at FROM twin_tokens WHERE user_id = ? ORDER BY id DESC", (g.user["id"],))
    return jsonify(tokens=tokens)


@app.route("/api/twin/ingest", methods=["POST"])
def api_twin_ingest():
    """ESP32 HTTP ingestion endpoint (Bearer device token OR logged-in browser session)."""
    uid, tok_id = None, None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        raw_tok = auth[7:].strip()
        if raw_tok:
            thash = hashlib.sha256(raw_tok.encode()).hexdigest()
            row = query("SELECT id, user_id FROM twin_tokens WHERE token_hash = ?", (thash,), one=True)
            if row:
                uid, tok_id = row["user_id"], row["id"]
    elif g.user:
        uid = g.user["id"]
    if uid is None:
        return jsonify(error="Unauthorized. Provide a valid ESP32 Bearer token."), 401
    if not _hit("twin_ingest", uid, 720, 3600):
        return jsonify(error="Ingestion rate limit reached."), 429

    b = request.get_json(silent=True) or {}
    dev_note = str(b.get("device_name", "ESP32")).strip()[:80]
    saved = []
    mapping = [("heart_rate", b.get("hr", b.get("heart_rate"))),
               ("spo2", b.get("spo2")),
               ("temperature", b.get("temp", b.get("temperature"))),
               ("resp_rate", b.get("rr", b.get("resp_rate")))]
    for metric, val in mapping:
        if val is not None:
            ok, _ = _save_sensor_reading(uid, metric, val, None, b.get("measured_at"), "esp32", dev_note)
            if ok:
                saved.append(metric)
    if b.get("bp_sys") is not None and b.get("bp_dia") is not None:
        ok, _ = _save_sensor_reading(uid, "bp", b.get("bp_sys"), b.get("bp_dia"), b.get("measured_at"), "esp32", dev_note)
        if ok:
            saved.append("bp")
    if not saved:
        return jsonify(error="No valid sensor values found (supported: hr, spo2, temp, rr, bp_sys/bp_dia)."), 400
    if tok_id is not None:
        query("UPDATE twin_tokens SET last_seen = ? WHERE id = ?", (_utcnow(), tok_id), commit=True)
    _twin_notify(uid)
    return jsonify(ok=True, saved=saved)


@app.route("/api/twin/stream")
@login_required
def api_twin_stream():
    uid = g.user["id"]
    q = queue.Queue(maxsize=20)
    with _TWIN_LOCK:
        _TWIN_SUBS.setdefault(uid, []).append(q)

    def gen():
        try:
            yield "event: twin_ready\ndata: {\"ok\": true}\n\n"
            while True:
                try:
                    msg = q.get(timeout=20)
                    yield "event: twin_update\ndata: " + json.dumps(msg) + "\n\n"
                except queue.Empty:
                    yield ": keepalive\n\n"
        finally:
            with _TWIN_LOCK:                      # always unsubscribe, or dead connections pile up in memory
                subs = _TWIN_SUBS.get(uid, [])
                if q in subs:
                    subs.remove(q)
                if not subs:
                    _TWIN_SUBS.pop(uid, None)

    return app.response_class(gen(), mimetype="text/event-stream",
                              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
