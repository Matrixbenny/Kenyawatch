// Debug: time the stages of one real OTP-style send through the app mailer.
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const nodemailer = require("nodemailer");

const t = (label, fn) =>
  fn().then((r) => {
    console.log(`${label}: ${Date.now() - start} ms`);
    return r;
  });

const start = Date.now();

(async () => {
  const transporter = nodemailer.createTransport({
    host: process.env.SMTP_HOST,
    port: Number(process.env.SMTP_PORT || 587),
    secure: process.env.SMTP_SECURE === "true",
    auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
    logger: false,
  });

  await t("verify()", () => transporter.verify());

  const info = await t("sendMail()", () =>
    transporter.sendMail({
      from: process.env.MAIL_FROM,
      to: process.env.SMTP_USER,
      subject: "Kenya Watch timing test",
      text: "timing test",
      html: "<p>timing test</p>",
    }),
  );
  console.log("accepted:", info.accepted, "rejected:", info.rejected);
  console.log("TOTAL:", Date.now() - start, "ms");
  transporter.close();
})().catch((e) => {
  console.error("ERR", e.message);
  console.log("at", Date.now() - start, "ms");
});
