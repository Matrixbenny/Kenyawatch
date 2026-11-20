// Keeps the local MongoDB as an up-to-date mirror of MongoDB Atlas, the main
// database used by the live site and the development server.
//
//   node scripts/mirror-atlas.js          refresh the local mirror once
//   node scripts/mirror-atlas.js --force  also overwrite local-only documents
//
// Reads ATLAS_URI (source) and LOCAL_MONGO_URI (target) from backend/.env.
// Atlas is only read. The target must be a database on this computer, and the
// mirror stops instead of deleting any document that exists only locally, so
// nothing written to the local database by mistake is ever lost silently.
require('dotenv').config({ quiet: true, path: require('path').join(__dirname, '..', '.env') });
const { MongoClient } = require('mongodb');

const LOCAL_HOST = /^mongodb:\/\/(localhost|127\.0\.0\.1)(:\d+)?\//;

function stamp() {
  return new Date().toISOString().replace('T', ' ').slice(0, 19);
}

async function mirror({ force = false } = {}) {
  const source = process.env.ATLAS_URI;
  const target = process.env.LOCAL_MONGO_URI;
  if (!source || !target) throw new Error('Set ATLAS_URI and LOCAL_MONGO_URI in backend/.env.');
  if (!LOCAL_HOST.test(target)) throw new Error('LOCAL_MONGO_URI must point to a database on this computer.');

  const from = await MongoClient.connect(source, { serverSelectionTimeoutMS: 20000 });
  const to = await MongoClient.connect(target, { serverSelectionTimeoutMS: 5000 });
  try {
    const src = from.db();
    const dst = to.db();
    const names = (await src.listCollections({ type: 'collection' }).toArray())
      .map((c) => c.name)
      .filter((name) => !name.startsWith('system.'));

    // Safety check: anything that exists only in the local copy would be deleted.
    const localOnly = [];
    for (const name of names) {
      const atlasIds = new Set((await src.collection(name).distinct('_id')).map(String));
      const extra = (await dst.collection(name).distinct('_id')).filter((id) => !atlasIds.has(String(id)));
      if (extra.length) localOnly.push(`${name} (${extra.length})`);
    }
    if (localOnly.length && !force) {
      throw new Error(`Local-only documents found in ${localOnly.join(', ')}. They are not in Atlas, so the mirror stopped to keep them. ` +
        'Copy them to Atlas first, or re-run with --force to discard them.');
    }

    let total = 0;
    for (const name of names) {
      const docs = await src.collection(name).find().toArray();
      const out = dst.collection(name);
      await out.deleteMany({});
      for (let i = 0; i < docs.length; i += 500) await out.insertMany(docs.slice(i, i + 500), { ordered: false });
      for (const index of await src.collection(name).indexes()) {
        if (index.name === '_id_') continue;
        const { key, v, ns, ...options } = index;
        await out.createIndex(key, options).catch(() => {});
      }
      total += docs.length;
    }
    return { collections: names.length, documents: total };
  } finally {
    await from.close();
    await to.close();
  }
}

if (require.main === module) {
  mirror({ force: process.argv.includes('--force') })
    .then((r) => console.log(`[${stamp()}] mirror ok: ${r.collections} collections, ${r.documents} documents copied from Atlas.`))
    .catch((err) => {
      console.error(`[${stamp()}] mirror failed: ${err.message}`);
      process.exit(1);
    });
}

module.exports = { mirror };
