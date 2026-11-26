// Diagnose why the "send me my tracking reference" email is slow.
// Run: node _mail_timing.js
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const nodemailer = require("nodemailer");

const t = () => Number(process.hrtime.bigint() / 1000000n);

(async () => {
  console.log("SMTP_HOST :", process.env.SMTP_HOST);
  console.log("SMTP_PORT :", process.env.SMTP_PORT);
  console.log("SMTP_SECURE:", process.env.SMTP_SECURE);
  console.log("SMTP_USER :", process.env.SMTP_USER);
  console.log("MAIL_FROM :", process.env.MAIL_FROM);
  console.log("NODE_ENV  :", process.env.NODE_ENV);

  const start = t();
  const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    connectionTimeout: 20000,
    greetingTimeout: 20000,
    socketTimeout: 30000,
    logger: true,
    debug: false,
  });

  console.log("\n-- verify() (connect + auth) --");
  try {
    const t0 = t();
    await transporter.verify();
    console.log("verify OK in", t() - t0, "ms");
  } catch (e) {
    console.log("verify FAILED after", t() - start, "ms:", e.message);
    return;
  }

  const target =
    process.env.MAIL_FROM && /@/.test(process.env.MAIL_FROM)
      ? process.env.MAIL_FROM.replace(/^.*<|>$/g, "").trim()
      : process.env.SMTP_USER;
  console.log("\n-- sendMail() to", target, "--");

  const t0 = t();
  try {
    const info = await transporter.sendMail({
      from: process.env.MAIL_FROM,
      to: target,
      subject: "Kenya Watch timing test",
      text: "This is a pure SMTP speed test.",
    });
    console.log("sendMail OK in", t() - t0, "ms | messageId", info.messageId);
  } catch (e) {
    console.log("sendMail FAILED after", t() - t0, "ms:", e.message);
  }

  transporter.close();
  console.log("\nTOTAL:", t() - start, "ms");
})().catch((e) => console.error("ERR", e.message));
