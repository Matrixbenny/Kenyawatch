const path=require("path");
require("dotenv").config({path:path.join(__dirname,".env")});
const mongoose=require("mongoose");
(async()=>{
  await mongoose.connect(process.env.MONGO_URI,{serverSelectionTimeoutMS:15000});
  const users=await mongoose.connection.db.collection("users").find({},{projection:{email:1,role:1,isActive:1,status:1}}).toArray();
  console.log("USERS:",JSON.stringify(users,null,1));
  const cols=await mongoose.connection.db.listCollections().toArray();
  console.log("COLLECTIONS:",cols.map(c=>c.name).join(", "));
  const reports=await mongoose.connection.db.collection("reports").countDocuments();
  console.log("REPORTS:",reports);
  await mongoose.disconnect();
})().catch(e=>{console.error("ERR",e.message);process.exit(1)});
