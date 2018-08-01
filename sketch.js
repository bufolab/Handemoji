


const hands = [//'🤲',
//'👐',
//'🙌',
//'👏',
//'🤝',
//'👍',
//'👎',
//'👊',
'✊',
//'🤛',
//'🤜',
//'🤞',
//'✌️',
//'🤟',
//'🤘',
//'👌',
//'👈',
//'👉',
//'👆',
//'👇',
//'☝️',
//'✋',
'🤚'
//'🖐',
//'🖖',
//'👋',
//'🤙',
//'💪',
//'🖕'
];


var video;
const SNAPSHOT_PX = "32px";
var MAX_TRAIN_CAPTURE = 64;

var training = false;

var samples = [];

const CAP_SIZE = {w:224,h:224};

var errorMessage;
var label;
var model ;
var traininIterationLabel;
let containerImages ;
var divArray = [];
var fileInput;
var predictButton;
var captureButton;
var emojiPredictedDiv;

function setup() {

var constraints = {
	audio: false,
	video: {
		mandatory: {
			minWidth: CAP_SIZE.w,
			minHeight: CAP_SIZE.h
		},
		optional: [{ maxFrameRate: 10 }]
	}
};
video = createCapture(constraints);
  //video.elt.width = CAP_SIZE.w;
  //video.elt.height = CAP_SIZE.h;
  video.style("margin","auto");
  video.parent("left-container");
  captureButton = createButton("Capture").mousePressed(prepareTrain);
  let trainButton = createButton("Start Training").mousePressed(startTrain);
  let predictButton = createButton("Predecit").mousePressed(predict);

  captureButton.parent("left-container");
  trainButton.parent("left-container");
  predictButton.parent("left-container");

  containerImages = createDiv();

  emojiPredictedDiv = createDiv();
  emojiPredictedDiv.position(video.position().x, video.position().y);
  emojiPredictedDiv.style("z-index","10");
  emojiPredictedDiv.style("font-size","200px");
  emojiPredictedDiv.style("opacity","0.5");
  emojiPredictedDiv.elt.innerHTML = ""
  errorMessage = createP("Please select a hand you want to train");
  errorMessage.hide();


  for(let i=0;i<hands.length;i++){
  	let hand = hands[i];
  	let button = createButton(hand);
  	button.style("margin","20px");
  	button.mousePressed(handSelected);
  	button.parent("target-container");

  	divImageContainer = createDiv();
  	divImageContainer.style("margin","20px");
  	divImageContainer.style("background-color","grey");
  	divImageContainer.hide();
  	divImageContainer.parent("right-container");
  	divArray[hand] = divImageContainer;
  }

	//presetup the model
	buildModel();
}

var m; //my machine
async function buildModel(){
	m = new Machine();
	m.build(hands.length);
	model = m.model;
}


//check the removeAlphaNormalize is working. We paint the processed image in a canvas
var test = function(){
	var img = removeAlphaNormalize(samples[0].data);
	loadPixels();
	for(let i = 0;i<CAP_SIZE.w*CAP_SIZE.h;i++){
		let c = color(img[i*3]*127, img[i*3+1]*127, img[i*3+2]*127);
		//console.log(aaa[i*3]*255 +""+ aaa[i*3+1]*255+""+ aaa[i*3+2]*255);
		set(i%CAP_SIZE.w,i/CAP_SIZE.w, c) ;
	}
	updatePixels();
}

var prepareTrain = async function(){
	
	if(training) return;
	training = true;

	if(label === undefined){
		errorMessage.show();
		training = false;
		return;
	}

	errorMessage.hide();

	for(let i=0;i<MAX_TRAIN_CAPTURE;i++){
		setTimeout(function(){
			let data = getCapture(i);
				//add all the training images the user want
				samples.push(data);
				if(i == MAX_TRAIN_CAPTURE -1){
					training = false;
				} 
			},50);
	}

}

var mtlabel ;
var getCapture = function(){
	let snapshotcanvas = video.get().canvas;
	let imagedata = snapshotcanvas.getContext('2d').getImageData(0,0,CAP_SIZE.w,CAP_SIZE.h);

	//show image
	let dataurl = snapshotcanvas.toDataURL("image/png");
	let img = createImg(dataurl);
	img.attribute("width", SNAPSHOT_PX);
	img.attribute("height", SNAPSHOT_PX );
	img.id("ImgSample");
	img.parent(divArray[mtlabel]);

	let data = removeAlphaNormalize(imagedata.data);

	//assign the label
	data.label = label;

	return data;

}

var handSelected = function(){
	errorMessage.hide();
	mtlabel = this.elt.innerHTML;
	label = hands.indexOf(this.elt.innerHTML);
	for (key in divArray) {
		let div = divArray[key];
		if(div === divArray[mtlabel]) div.show();
		else div.hide();
	}

	captureButton.elt.innerHTML ="Capture " + mtlabel;
}


function draw(){
	//nothing for the moment.
}


/*The tensorflow part*/

const PERCENTAGE_TEST = 20; //20% of all the input will be for testing
const TRAIN_BATCHES = 10;
const BATCH_SIZE = MAX_TRAIN_CAPTURE*2;

var trainIndices ;
var tensorImageSamples = [];
var tensorLabels = [];



function removeAlphaNormalize(imagedata){
	let imageBuffer = new Float32Array(CAP_SIZE.w*CAP_SIZE.h*3);
	for(let j = 0; j<CAP_SIZE.w*CAP_SIZE.h;j++) {
		let outIndex = j*3;
		let sourceIndex = j*4;
		imageBuffer[outIndex]=(imagedata[sourceIndex]/127 )-1; //R
		imageBuffer[outIndex+1]=((imagedata[sourceIndex+1]/127)-1); //G
		imageBuffer[outIndex+2]=((imagedata[sourceIndex+2]/127)-1); //B
	}

	return imageBuffer;
}


var isTraining = false;
var startTrain = async function() {

	if(isTraining) return;
	isTraining = true;

	try{

		//let trainIndices = tf.util.createShuffledIndices(samples.length);
		
		var labels = []
		myXs = [];
		for (var i =  0; i<samples.length; i++) {
			console.log(tf.memory().numBytes);
			//let index = trainIndices[i]; //shuffled in model.fit
			imagedata = samples[i];
			myXs.push(m.getTensorXs(imagedata));
		    //myXs.push( tf.tidy(() => { let t = createTensor(imagedata); return mobilenet.predict(t); }  ) ) ;
		    labels.push(imagedata.label);
		}
		
		let tfLabels = tf.tensor1d(labels, 'int32');
		//normalize the data so whatever the int of an emoji we normalize it between 0 and 1
		ys = tf.oneHot(tfLabels, hands.length).cast('int32');
		var xs = tf.concat(myXs);

		tfLabels.dispose();
		myXs.forEach(function(v){ v.dispose();});

		if (xs == null) {
			throw new Error('Add some examples before training!');
		}

	   //if (!(samples.length > 0)) {
	   // throw new Error(
	   //     `Batch size is 0 or NaN. Please choose a non-zero fraction.`);
	   // }

	   // Train the model! Model.fit() will shuffle xs & ys so we don't have to.
	   m.train(xs, ys);

	}catch(e){
		console.error(e);
		isTraining = false;
	}

}


var predict = async function () {

	let snapshotcanvas = video.get().canvas;
	let imagedata = snapshotcanvas.getContext('2d').getImageData(0,0,CAP_SIZE.w,CAP_SIZE.h);
	let img = removeAlphaNormalize(imagedata.data);

	const predictedClass = m.predict(m.getTensorXs(img));

	const classId =  predictedClass[0];
    //predictedClass.dispose();

    emojiPredictedDiv.elt.innerHTML = hands[classId];
    console.log(classId);
}
//}




