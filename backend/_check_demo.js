// One-shot demo verification: sign each demo account in against the live API.
const accounts = [
  ["amina@example.com", "ReviewerPass123!"],
  ["agencyofficer@kenyawatch.local", "AgencyPass123!"],
  ["admin@kenyawatch.local", "AdminPass123!"],
];

const BASE = "http://localhost:5000";

async function req(path, opts = {}) {
  const r = await fetch(BASE + path, opts);
  let body = null;
  try {
    body = await r.json();
  } catch {
    body = text;
  }
  return { status: r.status, body, cookie: r.headers.get("set-cookie") };
}

(async () => {
  const health = await req("/api/health");
  console.log("HEALTH", health.status, JSON.stringify(health.body));

  for (const [email, password] of accounts) {
    const r = await req("/api/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    const u = r.body && r.body.user;
    console.log(
      "LOGIN",
      email,
      "->",
      r.status,
      u
        ? `${u.name} [${u.role}] redirect=${r.body.redirect}`
        : JSON.stringify(r.body),
    );
    if (u && r.cookie) {
      const cookie = r.cookie.split(";")[0];
      for (const p of [
        "/api/reports",
        "/api/reports/categories",
        "/api/reports/queue",
        "/api/agency-portal/me",
        "/api/agency-portal/cases",
      ]) {
        const g = await req(p, { headers: { cookie } });
        const n = Array.isArray(g.body)
          ? g.body.length
          : (
              g.body &&
              (g.body.items || g.body.cases || g.body.reports || g.body.data)
            )?.length;
        console.log("   GET", p, g.status, "count=", n ?? "-");
      }
    }
  }
})();
