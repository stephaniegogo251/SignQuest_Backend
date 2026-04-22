var express = require("express");
var app = express(); //create instance of express.js
var path = require("path");
var fs = require("fs");
var dotenv = require('dotenv');
const port = process.env.PORT || 3000;
dotenv.config();
const cors = require('cors');

//config express.js
//cors middleware - applied to all requests to allow resources to be shared across diff domains
app.use ((req,res,next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader("Access-Control-Allow-Credentials", "true");
    res.setHeader("Access-Control-Allow-Methods", "GET,HEAD,OPTIONS,POST,PUT");
    res.setHeader("Access-Control-Allow-Headers", "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers");
 
    next();
});
app.use(cors());

//built-in express middleware to parse incoming json requests
app.use(express.json({ limit: '10mb'}));

//logging middleware
app.use(function(req, res, next){
    //logs request method, path, query and timestamp
    console.log(`${req.method} ${req.path} ${JSON.stringify(req.query)} time: ${new Date()}`);
    next();
});

//middleware to serve static files
app.use('/static', express.static(path.join(__dirname, 'static')));

//middleware to handle and send static files
app.use(function(req, res, next){
    var filePath = path.join(__dirname, "static", req.url);
    fs.stat(filePath, function(err, fileInfo){
        if (err){
            next();
            return;
        }
        if(fileInfo.isFile())
            res.sendFile(filePath);
        else
            next();
    });
});

const MongoClient = require('mongodb').MongoClient; //import mongoclient(object part of mongodb driver) to talk to mongo server
const uri = process.env.MONGODB_URI; //mongodb connection uri

let db;
//connect to mongodb
MongoClient.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true })
  .then(client => {
    db = client.db('SignQuest');  //connect to specific database
    console.log("Connected to database!");

    app.listen(port, () => {
        console.log(`Server running on ${port}`);
    });
})
.catch(err => console.error("Failed to connect to mongodb", err));

//default route for root path
app.get('/', (req, res, next) => {
    res.send('Select a collection, e.g., /collection/messages');
});

//route paramter middleware to attach collection to the request object for all other request handlers
app.param('collectionName', (req, res, next, collectionName) => {
    if (!db) return res.status(503).send({ msg: "Database not ready" });
    req.collection = db.collection(collectionName);
    return next();
});

//get route to return lessons collection as an array
app.get('/collection/:collectionName', (req, res, next) => {
    req.collection.find({}).toArray((e, results) => {
        if (e) return next(e);
        res.send(results);
    }); 
});

//post route to add order to orders collection
app.post('/collection/:collectionName', (req, res, next) => {
    req.collection.insertOne(req.body, (e, results) => {
        if (e) return next(e);
        res.send(results.ops);
    });
});

//get route to search for specific lessons from collection
app.get('/collection/:collectionName/search', (req, res, next) => {
    const query = req.query.q;
    const searchPattern = new RegExp(query, 'i'); //case-insensitive regex pattern
    let searchQuery = {
        $or: [
            {signName: {$regex: searchPattern}},
            {signAName: {$regex: searchPattern}},
        ]
    };

    // let numQuery = 0;
    // if (!isNaN(parseInt(query))) {
    //     numQuery = parseInt(query);
    //     searchQuery = {
    //     $or: [
    //         {name: {$regex: searchPattern}},
    //         {location: {$regex: searchPattern}},
    //         {price: numQuery},
    //         {availableSeats:numQuery}
    //     ]
    // };
    // }

    req.collection.find(searchQuery).toArray((e, results) => {
        if (e) return next(e);
        res.send(results);
    }); 

});

const ObjectID = require('mongodb').ObjectID;

//put route to update available spaces of a lesson
// app.put('/collection/:collectionName/:id', (req, res, next) => {
//     req.collection.updateOne(
//         {_id: new ObjectID(req.params.id)},
//         {$set: req.body},
//         {safe: true, multi: false},
//         (e, result) => {
//             if (e) return next(e)
//             res.send((result.result.n === 1) ? {msg: 'success'} : {msg: 'error'});
//         });
// });

app.put('/collection/:collectionName/:id', (req, res, next) => {
    req.collection.updateOne(
        { _id: new ObjectID(req.params.id) },
        { $set: req.body },
        (e, result) => {
            if (e) return next(e);
            res.send(result.modifiedCount === 1 ? { msg: 'success' } : { msg: 'error' });
        }
    );
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});