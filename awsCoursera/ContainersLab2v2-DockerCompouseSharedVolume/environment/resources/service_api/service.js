var 
	express = require("express"),
	fs = require("fs"),
	bodyParser = require("body-parser"),
	cors = require("cors"),
	config = {
	    name_str: "service",
	    port_int: 3000,
	    host_str: "0.0.0.0"
	},
	app = express();
app.use(bodyParser.json());
app.use(cors());
app.get("/", (req, res) => {
	res.status(200).send("hello this is the root of the service api");
});

var 
	g_payload = {
		cities_arr: ["vegas", "chicago"]
	};

var shared_file_path_str = "/my_amazing_shared_folder/service_data.json";

if(fs.existsSync(shared_file_path_str)) {
	console.log("The file exists, getting info");
	var file_data = fs.readFileSync(shared_file_path_str);
	// console.log(file_data.toString());
	g_payload = JSON.parse(file_data.toString());
}

//IF we are using the names volume overrider this again

shared_file_path_str = "/contains_your_service_area_data/my-data/service_data.json";

if(fs.existsSync(shared_file_path_str)) {
	console.log("The named volume service file exists, getting info");
	file_data = fs.readFileSync(shared_file_path_str);
	g_payload = JSON.parse(file_data.toString());//replace
}
app.get("/get-service-area", (req, res) => {
	res.status(200).json(g_payload);
	console.log("returning cities");
});
app.listen(config.port_int, config.host_str, (e)=> {
    if(e) {
        throw new Error("Internal Server Error");
    }
    console.log("Running the " + config.name_str + " app");
});