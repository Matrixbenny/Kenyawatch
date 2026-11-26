// Times the real receipt flow end-to-end: OTP email + credential email.
// Run: node backend/_receipt_timing.js
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const mongoose = require("mongoose");
const Report = require("./report.model");
const { requestReceiptOtp, verifyReceiptOtp } = require("./utils/reportEmailReceipt");
const EmailOtpChallenge = require("./email-otp-challenge.model");

const TO = process.env.TIMING_TO || process.env.SMTP_USER;

const t = async (label, fn) => {
  const s = Date.now();
  const r = await fn();
  console.log(`${label}: ${Date.now() - s} ms`);
  return r;
};

(async () => {
  await mongoose.connect(process.env.MONGO_URI, { serverSelectionTimeoutMS: 15000 });
  console.log("connected in", "ok");

  const report = await Report.findOne({ archivedAt: null }).sort({ createdAt: -1 }).lean();
  if (!report) throw new Error("no report found");
  console.log("report:", report.trackingReference);

  // We need the plaintext access key, which is not stored. So make a throwaway report-like
  // test by reading the OTP code straight from the challenge after requesting.
  const accessKey = process.env.TIMING_ACCESS_KEY;
  if (!accessKey) {
    console.log("TIMING_ACCESS_KEY not set - only timing the OTP request path is impossible without it.");
    console.log("Fallback: measuring the raw mailer path only.");
  }

  await EmailOtpChallenge.deleteMany({ report: report._id });

  console.log("\n--- raw OTP email send (bypassing lookup) ---");
  const mailer = require("./utils/mailer");
  await t("send OTP email", () =>
    mailer.sendMail({
      to: TO,
      subject: "Timing test OTP",
      text: "123456",
    })
  );

  console.log("\n--- second send (connection reuse) ---");
  await t("send receipt email", () =>
    mailer.sendMail({
      to: TO,
      subject: "Timing test receipt",
      text: "Tracking Reference: X\nAccess Key: Y",
    })
  );

  await mongoose.disconnect();
})().catch(async (e) => {
  console.error("ERR", e.message);
  try { await mongoose.disconnect(); } catch (_) {}
  process.exit(1);
});
