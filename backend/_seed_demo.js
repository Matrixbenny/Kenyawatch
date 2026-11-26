// One-shot demo seeding: creates/repairs the three demonstration accounts.
const path = require("path");
require("dotenv").config({ path: path.join(__dirname, ".env") });
const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const { AGENCIES, slugify } = require("./utils/agencies");

const DEMO = [
  {
    name: "Amina Wanjiku (Reviewer)",
    email: "amina@example.com",
    password: "ReviewerPass123!",
    role: "reviewer",
  },
  {
    name: "EACC Liaison Officer",
    email: "agencyofficer@kenyawatch.local",
    password: "AgencyPass123!",
    role: "agency",
  },
  {
    name: "Kenya Watch Administrator",
    email: "admin@kenyawatch.local",
    password: "AdminPass123!",
    role: "admin",
  },
];

(async () => {
  await mongoose.connect(process.env.MONGO_URI, {
    serverSelectionTimeoutMS: 15000,
  });
  const users = mongoose.connection.db.collection("users");
  const eacc = AGENCIES.find((a) => /EACC/.test(a.name));
  console.log("EACC slug:", slugify(eacc.name));

  for (const d of DEMO) {
    const existing = await users.findOne({ email: d.email });
    const hash = await bcrypt.hash(d.password, 11);
    const base = {
      name: d.name,
      email: d.email,
      role: d.role,
      isActive: true,
      emailNotificationsOptIn: false,
      createdAt: new Date(),
    };
    if (d.role !== "admin") base.password = hash;

    if (d.role === "agency") {
      base.agencySlug = slugify(eacc.name);
      base.agencyApplicationStatus = "approved";
      base.agencyVerificationReference = "DEMO-2026-EACC-001";
      base.agencyVerifiedAt = new Date();
      base.agencyInvitationAcceptedAt = new Date();
      base.agencyApplicationStaffNumber = "EACC-DEMO-01";
      base.agencyApplicationJobTitle = "Investigations Officer";
      base.agencyApplicationDutyStation = "Nairobi";
    }

    if (existing) {
      const patch = { ...base };
      if (d.role !== "admin") patch.password = hash;
      await users.updateOne({ _id: existing._id }, { $set: patch });
      console.log("updated", d.email, "role=", d.role);
    } else {
      const r = await users.insertOne(base);
      console.log(
        "created",
        d.email,
        "role=",
        d.role,
        "id=",
        String(r.insertedId),
      );
    }
  }

  const all = await users
    .find(
      { email: { $in: DEMO.map((d) => d.email) } },
      { projection: { email: 1, role: 1, isActive: 1 } },
    )
    .toArray();
  console.log("ACCOUNTS:", JSON.stringify(all, null, 1));
  console.log(
    "REPORTS:",
    await mongoose.connection.db.collection("reports").countDocuments(),
  );
  await mongoose.disconnect();
})().catch((e) => {
  console.error("ERR", e.message);
  process.exit(1);
});
