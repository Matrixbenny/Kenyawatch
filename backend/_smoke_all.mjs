// One-shot end-to-end smoke test for the Kenya Watch demo.
// Usage: node _smoke_all.mjs [baseUrl]
const BASE = process.argv[2] || "http://localhost:5000";

let pass = 0;
let fail = 0;

function ok(label, detail = "") {
  pass += 1;
  console.log(`  PASS  ${label}${detail ? "  -> " + detail : ""}`);
}
function bad(label, detail = "") {
  fail += 1;
  console.log(`  FAIL  ${label}${detail ? "  -> " + detail : ""}`);
}

async function call(path, { method = "GET", body, token, form, raw } = {}) {
  const headers = {};
  if (token) headers.authorization = `Bearer ${token}`;
  let payload;
  if (form) {
    payload = form;
  } else if (body !== undefined) {
    headers["content-type"] = "application/json";
    payload = JSON.stringify(body);
  }
  const res = await fetch(BASE + path, { method, headers, body: payload });
  const text = await res.text();
  let json = null;
  try {
    json = JSON.parse(text);
  } catch {
    json = null;
  }
  return { status: res.status, json, text, headers: res.headers };
}

// Prefer the token the API returns in the body; fall back to the cookie.
function tokenOf(r) {
  const b = r.json || {};
  return b.token || b.accessToken || b.jwt || null;
}

async function login(email, password) {
  const r = await call("/api/auth/login", {
    method: "POST",
    body: { email, password },
  });
  return { r, token: tokenOf(r), body: r.json };
}

(async () => {
  console.log(`\n=== Kenya Watch smoke test against ${BASE} ===\n`);

  // ---------------------------------------------------------------- 1. health
  console.log("1. Health & model");
  const health = await call("/api/health");
  if (health.status === 200)
    ok("GET /api/health", JSON.stringify(health.json).slice(0, 160));
  else bad("GET /api/health", `status ${health.status}`);

  const cats = await call("/api/reports/categories");
  if (cats.status === 200 && Array.isArray(cats.json?.categories))
    ok(
      "GET /api/reports/categories",
      `${cats.json.categories.length} categories`,
    );
  else bad("GET /api/reports/categories", `status ${cats.status}`);

  const agencies = await call("/api/reports/agency-partners");
  if (agencies.status === 200)
    ok(
      "GET /api/reports/agency-partners",
      JSON.stringify(agencies.json).slice(0, 120),
    );
  else bad("GET /api/reports/agency-partners", `status ${agencies.status}`);

  const liveStats = await call("/api/stats/public");
  if (liveStats.status === 200)
    ok("GET /api/stats/public", JSON.stringify(liveStats.json).slice(0, 160));
  else bad("GET /api/stats/public", `status ${liveStats.status}`);

  // ------------------------------------------------ 2. anonymous report path
  console.log("\n2. Anonymous citizen submission (Path A)");
  const cap = await call("/api/reports/captcha-challenge");
  const challenge = cap.json || {};
  const tokenMatch = JSON.stringify(challenge).match(/"token"\s*:\s*"([^"]+)"/);
  const answerMatch = JSON.stringify(challenge).match(
    /"answer"\s*:\s*"?([^"]+)"?/,
  );
  const challengeId = challenge.challengeId || challenge.id || null;
  const challengeToken = tokenMatch ? tokenMatch[1] : null;
  const answer = answerMatch ? answerMatch[1] : null;
  console.log(
    "  captcha payload keys:",
    Object.keys(challenge).join(", "),
    challengeId ? "| id ok" : "| NO ID",
    challengeToken ? "| token ok" : "| NO TOKEN",
    answer ? "| answer in payload" : "| no answer",
  );

  const form = new FormData();
  form.append("title", "Smoke test: bribe demanded at a police checkpoint");
  form.append(
    "description",
    "An officer at the checkpoint on Ngong Road asked for kitu kidogo before he would let the matatu pass. He said two hundred shillings would settle it and threatened to impound the vehicle otherwise.",
  );
  form.append("category", challenge.defaultCategory || "bribery");
  form.append("county", "Nairobi");
  form.append("location", "Ngong Road");
  form.append("anonymous", "true");
  form.append("consent", "true");
  if (challengeId) form.append("challengeId", challengeId);
  if (challengeToken) form.append("challengeToken", challengeToken);
  if (answer) form.append("captchaAnswer", answer);

  const submit = await call("/api/reports", { method: "POST", form });
  const ref = submit.json?.reference || submit.json?.trackingCode || null;
  const key = submit.json?.accessKey || submit.json?.trackingKey || null;
  if (submit.status === 201 || submit.status === 200)
    ok(
      "POST /api/reports (anonymous)",
      `status ${submit.status} ref=${ref} key=${key ? "issued" : "missing"}`,
    );
  else
    bad(
      "POST /api/reports (anonymous)",
      `status ${submit.status} ${submit.text.slice(0, 300)}`,
    );

  if (ref) {
    const trackOk = await call("/api/reports/track", {
      method: "POST",
      body: { reference: ref, accessKey: key },
    });
    if (trackOk.status === 200)
      ok("POST /api/reports/track (correct key)", `status ${trackOk.status}`);
    else
      bad("POST /api/reports/track (correct key)", `status ${trackOk.status}`);

    const trackBad = await call("/api/reports/track", {
      method: "POST",
      body: { reference: ref, accessKey: "WRONG-KEY-0000" },
    });
    if ([400, 401, 403, 404].includes(trackBad.status))
      ok(
        "POST /api/reports/track (wrong key refused)",
        `status ${trackBad.status} - neutral denial`,
      );
    else
      bad(
        "POST /api/reports/track (wrong key refused)",
        `status ${trackBad.status}`,
      );
  }

  // ------------------------------------------------------------- 3. reviewer
  console.log("\n3. Reviewer (Amina)");
  const rev = await login("amina@example.com", "ReviewerPass123!");
  if (rev.r.status === 200)
    ok(
      "POST /api/auth/login reviewer",
      `role=${rev.body?.user?.role || rev.body?.role || "?"}`,
    );
  else
    bad(
      "POST /api/auth/login reviewer",
      `status ${rev.r.status} ${rev.r.text.slice(0, 200)}`,
    );

  if (rev.body?.mfaRequired)
    console.log(
      "  NOTE: reviewer requires a second step; demo must complete it in the UI.",
    );

  if (rev.token) {
    const queue = await call("/api/reports/queue", { token: rev.token });
    if (queue.status === 200)
      ok(
        "GET /api/reports/queue",
        `${Array.isArray(queue.json?.reports) ? queue.json.reports.length : "?"} in queue`,
      );
    else bad("GET /api/reports/queue", `status ${queue.status}`);

    const escal = await call("/api/reports/escalations", { token: rev.token });
    console.log(`  INFO  GET /api/reports/escalations -> ${escal.status}`);

    if (ref) {
      const created = (queue.json?.reports || []).find(
        (x) => (x.reference || x.trackingCode) === ref,
      );
      if (created) {
        const id = created._id || created.id;
        const sug = await call(`/api/reports/${id}/ml-category-suggestion`, {
          token: rev.token,
        });
        if (sug.status === 200)
          ok(
            "GET /api/reports/:id/ml-category-suggestion",
            `${sug.json?.category || sug.json?.suggestion?.category || "?"} conf=${sug.json?.confidence ?? sug.json?.suggestion?.confidence ?? "?"}`,
          );
        else
          bad(
            "GET /api/reports/:id/ml-category-suggestion",
            `status ${sug.status}`,
          );
      } else {
        console.log(
          "  INFO  new report not visible in queue (may need reviewer triage/refresh).",
        );
      }
    }
  }

  // --------------------------------------------------- 4. agency officer portal
  console.log("\n4. Agency officer (Agency Portal)");
  const ag = await login("agencyofficer@kenyawatch.local", "AgencyPass123!");
  if (ag.r.status === 200)
    ok("POST /api/auth/login agency officer", `status 200`);
  else
    bad(
      "POST /api/auth/login agency officer",
      `status ${ag.r.status} ${ag.r.text.slice(0, 200)}`,
    );

  if (ag.token) {
    const me = await call("/api/agency-portal/me", { token: ag.token });
    if (me.status === 200)
      ok("GET /api/agency-portal/me", JSON.stringify(me.json).slice(0, 140));
    else bad("GET /api/agency-portal/me", `status ${me.status}`);

    const cases = await call("/api/agency-portal/cases", { token: ag.token });
    if (cases.status === 200) {
      const n = Array.isArray(cases.json?.cases)
        ? cases.json.cases.length
        : "?";
      ok("GET /api/agency-portal/cases", `${n} referrals visible`);
      const first = (cases.json?.cases || [])[0];
      if (first) {
        const id = first._id || first.id;
        const upd = await call(`/api/agency-portal/cases/${id}/updates`, {
          method: "POST",
          token: ag.token,
          body: {
            stage: "acknowledged",
            note: "Smoke test: case acknowledged by the receiving institution.",
          },
        });
        if ([200, 201].includes(upd.status))
          ok(
            "POST /api/agency-portal/cases/:id/updates",
            `status ${upd.status}`,
          );
        else
          bad(
            "POST /api/agency-portal/cases/:id/updates",
            `status ${upd.status} ${upd.text.slice(0, 200)}`,
          );
      }
    } else bad("GET /api/agency-portal/cases", `status ${cases.status}`);

    // Boundaries: an agency officer must not reach reviewer or admin surfaces.
    for (const guarded of [
      "/api/reports/queue",
      "/api/users",
      "/api/reports/archived",
    ]) {
      const g = await call(guarded, { token: ag.token });
      if ([401, 403].includes(g.status))
        ok(`boundary: agency blocked from ${guarded}`, `status ${g.status}`);
      else
        bad(
          `boundary: agency blocked from ${guarded}`,
          `status ${g.status} (should be 403)`,
        );
    }
  }

  // ------------------------------------------------------------- 5. admin
  console.log("\n5. Admin");
  const adm = await login("admin@kenyawatch.local", "AdminPass123!");
  if (adm.r.status === 200) ok("POST /api/auth/login admin", `status 200`);
  else
    bad(
      "POST /api/auth/login admin",
      `status ${adm.r.status} ${adm.r.text.slice(0, 200)}`,
    );

  if (adm.token) {
    const users = await call("/api/users", { token: adm.token });
    if (users.status === 200)
      ok(
        "GET /api/users",
        `${Array.isArray(users.json?.users) ? users.json.users.length : "?"} accounts`,
      );
    else bad("GET /api/users", `status ${users.status}`);

    const archived = await call("/api/reports/archived", { token: adm.token });
    if (archived.status === 200) ok("GET /api/reports/archived", `status 200`);
    else bad("GET /api/reports/archived", `status ${archived.status}`);

    const esc = await call("/api/reports/escalations", { token: adm.token });
    if (esc.status === 200) ok("GET /api/reports/escalations", `status 200`);
    else bad("GET /api/reports/escalations", `status ${esc.status}`);

    // Boundaries: a reviewer must not reach as much as an admin.
    if (rev.token) {
      const rv = await call("/api/users", { token: rev.token });
      if ([401, 403].includes(rv.status))
        ok("boundary: reviewer blocked from /api/users", `status ${rv.status}`);
      else
        bad(
          "boundary: reviewer blocked from /api/users",
          `status ${rv.status} (should be 403)`,
        );
    }
  }

  // ------------------------------------------------------- 6. frontend build
  console.log("\n6. Frontend");
  try {
    const fe = await fetch("http://localhost:3000/");
    const html = await fe.text();
    if (fe.status === 200 && /<div id="root"|<script/.test(html))
      ok("GET http://localhost:3000/", `status 200, ${html.length} bytes`);
    else bad("GET http://localhost:3000/", `status ${fe.status}`);
  } catch (e) {
    bad("GET http://localhost:3000/", e.message);
  }

  console.log(`\n=== RESULT: ${pass} passed, ${fail} failed ===\n`);
  process.exit(fail ? 1 : 0);
})();
