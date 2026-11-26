// One-shot diagnostic: how long does each SMTP step take on this machine?
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const nodemailer = require("nodemailer");
const net = require("net");

const ms = (t) => Math.round(Number(process.hrtime.bigint() - t) / 1e6);
const mark = (t) => ms(t);

(async () => {
  console.log(
    "SMTP_HOST:",
    process.env.SMTP_HOST,
    "PORT:",
    process.env.SMTP_PORT,
  );

  // 1. DNS
  let t = process.hrtime.bigint();
  const dns = require("dns").promises;
  const addrs = await dns.lookup(process.env.SMTP_HOST, { all: true });
  console.log("dns lookup ms:", mark(t), addrs.map((a) => a.address).join(","));

  // 2. raw TCP connect
  t = process.hrtime.bigint();
  const sock = net.connect(
    Number(process.env.SMTP_PORT),
    process.env.SMTP_HOST,
  );
  await new Promise((res, rej) => {
    sock.once("connect", res);
    sock.once("error", rej);
    sock.setTimeout(15000, () => rej(new Error("tcp timeout")));
  });
  console.log("tcp connect ms:", mark(t));
  sock.destroy();

  // 3. nodemailer full handshake + verify
  const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    logger: false,
    debug: false,
  });
  t = process.hrtime.bigint();
  const ok = await transporter.verify();
  console.log("transporter.verify ms:", mark(t), ok);

  // 4. per-command log
  transporter.on("log", () => {});
  const verbose = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    logger: true,
    debug: true,
  });
  t = process.hrtime.bigint();
  const info = await verbose.sendMail({
    from: process.env.MAIL_FROM || '"Kenya Watch" <no-reply@kenyawatch.local>',
    to: process.env.SMTP_USER,
    subject: "Kenya Watch SMTP timing test",
    text: "timing test",
  });
  console.log("FULL sendMail ms:", mark(t), "messageId:", info.messageId);
  verbose.close();
  process.exit(0);
})().catch((e) => {
  console.error("ERR", e.message);
  process.exit(1);
});
