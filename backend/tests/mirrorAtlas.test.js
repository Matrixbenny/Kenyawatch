const test = require('node:test');
const assert = require('node:assert/strict');
const { MongoClient } = require('mongodb');
const { mirror } = require('../scripts/mirror-atlas');

// Two scratch databases on the local server stand in for Atlas and the mirror.
const BASE = (process.env.TEST_MONGO_URI || 'mongodb://localhost:27017/kenya_watch_test').replace(/\/[^/?]*(\?.*)?$/, '');
const SOURCE = `${BASE}/kenya_watch_test_mirror_src`;
const TARGET = `${BASE}/kenya_watch_test_mirror_dst`;

test('the local mirror copies Atlas, never deletes local-only data silently, and only writes locally', async (t) => {
  const saved = { ATLAS_URI: process.env.ATLAS_URI, LOCAL_MONGO_URI: process.env.LOCAL_MONGO_URI };
  process.env.ATLAS_URI = SOURCE;
  process.env.LOCAL_MONGO_URI = TARGET;
  const client = await MongoClient.connect(BASE);
  const src = client.db('kenya_watch_test_mirror_src');
  const dst = client.db('kenya_watch_test_mirror_dst');
  t.after(async () => {
    await src.dropDatabase();
    await dst.dropDatabase();
    await client.close();
    for (const [k, v] of Object.entries(saved)) { if (v === undefined) delete process.env[k]; else process.env[k] = v; }
  });
  await src.dropDatabase();
  await dst.dropDatabase();

  await src.collection('reports').insertMany([{ _id: 'a', title: 'one' }, { _id: 'b', title: 'two' }]);
  const first = await mirror();
  assert.equal(first.documents, 2);
  assert.equal(await dst.collection('reports').countDocuments(), 2);

  // A document that exists only locally stops the mirror and survives.
  await dst.collection('reports').insertOne({ _id: 'local-only', title: 'written locally' });
  await assert.rejects(mirror(), /Local-only documents found in reports \(1\)/);
  assert.equal(await dst.collection('reports').countDocuments({ _id: 'local-only' }), 1);

  // --force discards it on purpose.
  await mirror({ force: true });
  assert.equal(await dst.collection('reports').countDocuments({ _id: 'local-only' }), 0);

  // The target must be on this computer.
  process.env.LOCAL_MONGO_URI = 'mongodb+srv://user:pass@cluster0.example.mongodb.net/kenya_watch';
  await assert.rejects(mirror(), /must point to a database on this computer/);
});
