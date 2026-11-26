// Isolates what costs time inside one SMTP send.
// Run: node backend/_smtp_stages.js
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const nodemailer = require("nodemailer");

const TO = process.env.TIMING_TO || process.env.SMTP_USER;

function mk(opts) {
  return nodemailer.createTransport(opts);
}

(async () => {
  const base = {
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
  };

  console.log("--- A: default transport (what the app uses now) ---");
  let s = Date.now();
  const a = mk(base);
  await new Promise((res, rej) => a.verify((e) => (e ? rej(e) : res())));
  console.log("verify:", Date.now() - s, "ms");

  s = Date.now();
  await a.sendMail({
    from: process.env.MAIL_FROM,
    to: TO,
    subject: "stage A",
    text: "a",
  });
  console.log("send #1:", Date.now() - s, "ms");

  s = Date.now();
  await a.sendMail({
    from: process.env.MAIL_FROM,
    to: TO,
    subject: "stage A2",
    text: "a",
  });
  console.log("send #2 (pipelining):", Date.now() - s, "ms");

  console.log("\n--- B: pool + DNS warm (shared connection) ---");
  s = Date.now();
  const b = mk({
    ...base,
    pool: true,
    maxConnections: 1,
    maxMessages: 20,
    dnsTimeout: 5000,
  });
  await new Promise((res, rej) => b.verify((e) => (e ? rej(e) : res())));
  console.log("verify:", Date.now() - s, "ms");

  s = Date.now();
  await b.sendMail({
    from: process.env.MAIL_FROM,
    to: TO,
    subject: "stage B",
    text: "b",
  });
  console.log("send #1:", Date.now() - s, "ms");

  s = Date.now();
  await b.sendMail({
    from: process.env.MAIL_FROM,
    to: TO,
    subject: "stage B2",
    text: "b",
  });
  console.log("send #2 (reused conn):", Date.now() - s, "ms");

  b.close();
  a.close();
})().catch((e) => {
  console.error("ERR", e.message);
  process.exit(1);
});
