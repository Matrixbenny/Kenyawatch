// How close is the live database to being trainable on real reports?
//
// The readiness gate needs 450 adjudicated reports, at least 30 in every category.
// Those reports are produced slowly by the blinded two-reviewer workflow, so this
// script answers one question without exporting anything: how many reports would
// the exporter actually accept today?
//
// It uses the same eligibility rule as export-reviewed-dataset.js, so it cannot
// disagree with the export or with what the gate will later see.
//
//   node ml/scripts/label-progress.js
//   node ml/scripts/label-progress.js --json
//
// Reads only the categoryReview fields. No narrative text, evidence, identity or
// tracking reference is loaded or printed.

const path = require("path");
const mongoose = require("../../backend/node_modules/mongoose");
require("../../backend/node_modules/dotenv").config({
  path: path.resolve(__dirname, "../../backend/.env"),
});
const Report = require("../../backend/report.model");
const { CATEGORIES } = require("../../backend/utils/categories");

// Keep in step with ml/src/readiness.py
const MIN_ROWS = 450;
const MIN_PER_CATEGORY = 30;

// Identical to resolveApprovedLabel() in export-reviewed-dataset.js.
function approvedLabel(report) {
  const review = report.categoryReview || {};
  const category = review.adjudicatedCategory;
  const decision = review.adjudicatedDecision || (category ? "category" : null);
  const finalized =
    review.status === "agreed" || review.status === "adjudicated";
  if (finalized && decision === "category" && CATEGORIES.includes(category))
    return category;
  return null;
}

function bar(done, target, width = 24) {
  const filled = Math.min(width, Math.round((done / target) * width));
  return `${"#".repeat(filled)}${".".repeat(width - filled)}`;
}

async function main() {
  const asJson = process.argv.includes("--json");
  const uri = process.env.MONGO_URI || "mongodb://localhost:27017/kenya_watch";
  await mongoose.connect(uri, { serverSelectionTimeoutMS: 5000 });

  // Only the review block and archive flag are needed; never the narrative.
  const reports = await Report.find(
    {},
    { categoryReview: 1, archivedAt: 1, label: 1 },
  ).lean();

  const perCategory = Object.fromEntries(
    CATEGORIES.map((category) => [category, 0]),
  );
  let eligible = 0;
  let awaitingSecond = 0;
  let needsAdjudication = 0;
  let insufficient = 0;
  let unreviewed = 0;
  let archivedEligible = 0;

  for (const report of reports) {
    const label = approvedLabel(report);
    const status = report.categoryReview && report.categoryReview.status;
    if (label) {
      if (report.archivedAt) {
        archivedEligible += 1;
        continue;
      }
      eligible += 1;
      perCategory[label] += 1;
      continue;
    }
    if (status === "awaiting_second_review") awaitingSecond += 1;
    else if (status === "needs_adjudication") needsAdjudication += 1;
    else if (status === "insufficient_information") insufficient += 1;
    else unreviewed += 1;
  }

  const thin = CATEGORIES.filter(
    (category) => perCategory[category] < MIN_PER_CATEGORY,
  );
  const reportsNeeded = Math.max(0, MIN_ROWS - eligible);
  const whatIsBlocking = [];
  if (eligible < MIN_ROWS)
    whatIsBlocking.push(`${MIN_ROWS - eligible} more eligible reports`);
  if (thin.length)
    whatIsBlocking.push(`at least ${MIN_PER_CATEGORY} in: ${thin.join(", ")}`);

  const result = {
    database: uri.replace(/\/\/[^@]*@/, "//***@"),
    eligibleReports: eligible,
    requiredReports: MIN_ROWS,
    reportsStillNeeded: reportsNeeded,
    perCategory,
    minimumPerCategory: MIN_PER_CATEGORY,
    categoriesBelowMinimum: thin,
    inProgress: {
      awaitingSecondReview: awaitingSecond,
      needsAdminAdjudication: needsAdjudication,
      insufficientInformation: insufficient,
      notYetReviewed: unreviewed,
    },
    archivedButEligible: archivedEligible,
    blockedBy: whatIsBlocking.length
      ? whatIsBlocking
      : ["nothing: enough eligible reports for a real-data run"],
    trainableNow: eligible >= MIN_ROWS && thin.length === 0,
  };

  if (asJson) {
    console.log(JSON.stringify(result, null, 2));
    await mongoose.disconnect();
    return;
  }

  console.log(`Database: ${result.database}`);
  console.log(
    `\nEligible reports: ${eligible} / ${MIN_ROWS}   [${bar(eligible, MIN_ROWS)}]`,
  );
  console.log(
    reportsNeeded
      ? `Still needed: ${reportsNeeded}`
      : "Target number of reports reached.",
  );
  console.log("\nPer category (minimum " + MIN_PER_CATEGORY + "):");
  for (const category of CATEGORIES) {
    const count = perCategory[category];
    const flag = count < MIN_PER_CATEGORY ? "  <- below minimum" : "";
    console.log(
      `  ${category.padEnd(28)} ${String(count).padStart(4)}  [${bar(count, MIN_PER_CATEGORY)}]${flag}`,
    );
  }
  console.log("\nWork in progress:");
  console.log(`  waiting for a second reviewer : ${awaitingSecond}`);
  console.log(`  needs Admin adjudication      : ${needsAdjudication}`);
  console.log(`  judged insufficient to classify: ${insufficient}`);
  console.log(`  not reviewed yet              : ${unreviewed}`);
  if (archivedEligible)
    console.log(`  archived (excluded by the exporter): ${archivedEligible}`);
  console.log(
    "\nWould the exporter accept a real-data run today?",
    result.trainableNow ? "YES" : "NO",
  );
  console.log("Blocked by:", whatIsBlocking.join("; ") || "nothing");

  await mongoose.disconnect();
}

main().catch(async (error) => {
  console.error(`label-progress failed: ${error.message}`);
  try {
    await mongoose.disconnect();
  } catch {
    /* already closed */
  }
  process.exit(1);
});
