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
    db = client.db('SignQuest')}); //connect to specific database

//default route for root path
app.get('/', (req, res, next) => {
    res.send('Select a collection, e.g., /collection/messages');
});

//route paramter middleware to attach collection to the request object for all other request handlers
app.param('collectionName', (req, res, next, collectionName) => {
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
    req.collection.insert(req.body, (e, results) => {
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
            {name: {$regex: searchPattern}},
            {location: {$regex: searchPattern}},
        ]
    };

    let numQuery = 0;
    if (!isNaN(parseInt(query))) {
        numQuery = parseInt(query);
        searchQuery = {
        $or: [
            {name: {$regex: searchPattern}},
            {location: {$regex: searchPattern}},
            {price: numQuery},
            {availableSeats:numQuery}
        ]
    };
    }

    req.collection.find(searchQuery).toArray((e, results) => {
        if (e) return next(e);
        res.send(results);
    }); 

});

const ObjectID = require('mongodb').ObjectID;

//put route to update available spaces of a lesson
app.put('/collection/:collectionName/:id', (req, res, next) => {
    req.collection.updateOne(
        {_id: new ObjectID(req.params.id)},
        {$set: req.body},
        {safe: true, multi: false},
        (e, result) => {
            if (e) return next(e)
            res.send((result.result.n === 1) ? {msg: 'success'} : {msg: 'error'});
        });
});

const { spawn } = require('child_process');

// Use python3 for Render/Linux environments
const pythonProcess = spawn('python3', ['predict_script.py']);

// listens for errors
pythonProcess.stderr.on('data', (data) => {
    console.error(`Python Error: ${data}`);
});

let isPythonReady = false;

pythonProcess.stdout.on('data', (data) => {
    const message = data.toString().trim();
    
    if (message === "READY") {
        isPythonReady = true;
        console.log("models loaded!");
    }
});

app.post('/api/predict', (req, res) => {
    if (!isPythonReady) {
        return res.status(503).json({ error: "AI model is still loading, please wait 30 seconds." });
    }

    if (!pythonProcess || pythonProcess.killed) {
        return res.status(500).json({ error: "Python process is not running" });
    }

    const payload = JSON.stringify({
        path: req.body.path,
        keypoints: req.body.keypoints
    });

    // remove any old listeners
    pythonProcess.stdout.removeAllListeners('data');

    // set up one-time listener for specific request
    pythonProcess.stdout.once('data', (data) => {
        const message = data.toString().trim();
        console.log("Raw Python Output:", message);

        // Check if the message is actually JSON
        if (message.startsWith('{')) {
            try {
                const result = JSON.parse(message);
                console.log("SUCCESS: Sending result to frontend");
                return res.json(result); 
            } catch (e) {
                return res.status(500).json({ error: "JSON Parse Error", raw: message });
            }
        } else {
            console.warn("Python sent non-JSON info:", message);
            return res.status(500).json({ error: "Expected JSON but got something else", raw: message });
        }
    });

    pythonProcess.stdin.write(payload + '\n');
});

app.listen(port, () => {
    console.log(`Server running on port ${port}`);
});