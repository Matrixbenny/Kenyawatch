// One-shot SMTP speed test. Run: node _smtp_test.js you@example.com
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const nodemailer = require("nodemailer");

(async () => {
  const to = process.argv[2] || process.env.SMTP_USER;
  for (const port of [465, 587]) {
    const t0 = Date.now();
    const tr = nodemailer.createTransport({
      host: process.env.SMTP_HOST,
      port,
      secure: port === 465,
      auth: { user: process.env.SMTP_USER, pass: process.env.SMTP_PASS },
      connectionTimeout: 15000,
      greetingTimeout: 15000,
      socketTimeout: 20000,
    });
    try {
      await tr.verify();
      const tv = Date.now();
      console.log(`port ${port}: connected+auth in ${tv - t0} ms`);
      const s = Date.now();
      const info = await tr.sendMail({
        from: process.env.MAIL_FROM || process.env.SMTP_USER,
        to,
        subject: "Kenya Watch SMTP speed test",
        text: `Port ${port} test at ${new Date().toISOString()}`,
      });
      console.log(
        `port ${port}: sent in ${Date.now() - s} ms -> ${info.messageId}`,
      );
    } catch (e) {
      console.log(
        `port ${port}: FAILED after ${Date.now() - t0} ms -> ${e.code || ""} ${e.message}`,
      );
    }
    tr.close();
  }
})();
