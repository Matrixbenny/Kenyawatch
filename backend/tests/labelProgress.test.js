// Tests the counting rules in label-progress.js without needing a database.
//
// The real script queries MongoDB; this checks the one thing that can silently go
// wrong: whether a report counts as eligible, and how the in-progress buckets split.
const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("path");

const { CATEGORIES } = require("../../backend/utils/categories");

// Mirrors approvedLabel() in ml/scripts/label-progress.js and
// resolveApprovedLabel() in ml/scripts/export-reviewed-dataset.js.
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

function bucket(report) {
  const label = approvedLabel(report);
  if (label) return report.archivedAt ? "archived" : "eligible";
  const status = report.categoryReview && report.categoryReview.status;
  if (status === "awaiting_second_review") return "awaitingSecondReview";
  if (status === "needs_adjudication") return "needsAdjudication";
  if (status === "insufficient_information") return "insufficient";
  return "notYetReviewed";
}

test("two matching reviewer labels are eligible", () => {
  assert.equal(
    approvedLabel({
      categoryReview: { status: "agreed", adjudicatedCategory: "Bribery" },
    }),
    "Bribery",
  );
});

test("an Admin adjudication is eligible", () => {
  assert.equal(
    approvedLabel({
      categoryReview: {
        status: "adjudicated",
        adjudicatedDecision: "category",
        adjudicatedCategory: "Extortion",
      },
    }),
    "Extortion",
  );
});

test("insufficient information is never a category label", () => {
  assert.equal(
    approvedLabel({
      categoryReview: {
        status: "insufficient_information",
        adjudicatedDecision: "insufficient_information",
      },
    }),
    null,
  );
});

test("a disagreement still waiting for the Admin is not eligible", () => {
  assert.equal(
    approvedLabel({
      categoryReview: {
        status: "needs_adjudication",
        adjudicatedCategory: "Bribery",
      },
    }),
    null,
  );
});

test("an unreviewed report is not eligible", () => {
  assert.equal(approvedLabel({ categoryReview: {} }), null);
  assert.equal(approvedLabel({}), null);
});

test("a label outside the canonical taxonomy is not eligible", () => {
  assert.equal(
    approvedLabel({
      categoryReview: {
        status: "agreed",
        adjudicatedCategory: "Someone Typed This By Hand",
      },
    }),
    null,
  );
});

test("an archived eligible report is counted separately and never as trainable", () => {
  assert.equal(
    bucket({
      categoryReview: { status: "agreed", adjudicatedCategory: "Bribery" },
      archivedAt: new Date(),
    }),
    "archived",
  );
});

test("the in-progress buckets split as the reviewer would expect", () => {
  assert.equal(
    bucket({ categoryReview: { status: "awaiting_second_review" } }),
    "awaitingSecondReview",
  );
  assert.equal(
    bucket({ categoryReview: { status: "needs_adjudication" } }),
    "needsAdjudication",
  );
  assert.equal(
    bucket({ categoryReview: { status: "insufficient_information" } }),
    "insufficient",
  );
  assert.equal(bucket({ categoryReview: {} }), "notYetReviewed");
  assert.equal(bucket({}), "notYetReviewed");
});
