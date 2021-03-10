var 
	express = require("express"),
	fs = require("fs"),
	bodyParser = require("body-parser"),
	cors = require("cors"),
	config = {
	    name_str: "price",
	    port_int: 3000,
	    host_str: "0.0.0.0"
	},
	app = express();
app.use(bodyParser.json());
app.use(cors());
app.get("/", (req, res) => {
	res.status(200).send("hello this is the root of the price api");
});

var 
	g_payload = {
		price_arr: ["134.50", "455.99"]
	};

var shared_file_path_str = "/my_amazing_shared_folder/price_data.json";


if(fs.existsSync(shared_file_path_str)) {
	console.log("The price file exists, getting info");
	var file_data = fs.readFileSync(shared_file_path_str);
	g_payload = JSON.parse(file_data.toString());
}


//IF we are using the names volume overrider this again

shared_file_path_str = "/contains_your_price_data/my-data/price_data.json";

if(fs.existsSync(shared_file_path_str)) {
	console.log("The named volume price file exists, getting info");
	file_data = fs.readFileSync(shared_file_path_str);
	g_payload = JSON.parse(file_data.toString());//replace
}


app.get("/get-prices", (req, res) => {
	res.status(200).json(g_payload);
	console.log("returning prices");
});
app.listen(config.port_int, config.host_str, (e)=> {
    if(e) {
        throw new Error("Internal Server Error");
    }
    console.log("Running the " + config.name_str + " app");
});