var cx;
var cy;
var basicMenu;
var subMenu1;
var subMenu2;
var backgroundColor;
var numElements = 5;
var newColors;

function setup() {
  createCanvas(windowWidth, windowHeight);
  colorMode(HSB, 255);
  frameRate(60);
  background(0);
  noStroke();
  cx 		= width/2;
  cy 		= height/2;
  strokeCap(ROUND);
  strokeJoin(ROUND);

  //Configurations for portrait / landscape / square displays
  if (height > width) {
    dBias 	= cx;
  } else if (width > height) {
    dBias 	= cy;
  } else if (width == height) {
    dBias	= cx;
  }

  basicMenu = new dropMenu(
	cx,						//x-coordinate of menu
	height-cy/2,			//y-coordinate of menu
	1.0,					//size of menu (1.0)
	PI,						//what degree (and direction) menu icon rotates when opening
	width,					//width of display
	height,					//height of display
	dBias,					//for dpi scaling
	2665,						//duration, in frames, of animations. 0=animations disabled
	numElements,			//number of elements (length of list) for the menu to handle
	true,					//this menu must be clicked to open
	'UP',					//this menu opens upwards
	);
  backgroundColor = color(0,0,128);
  newColors = new SinglyList();
  for (var i = 0; i < numElements; i++)
	newColors.add(color(i*(256/numElements), 128, 128));
}

function draw() {
  background(backgroundColor);
  basicMenu.drawMenu();
  if((basicMenu.isOpen())) {
	var col;
	var c;
	var rs = basicMenu.getRS();
	if (rs > 0){
	  col = newColors.searchNodeAt(basicMenu.getR());
	  c = col.data;
	  fill(c);
	  ellipse(
		basicMenu.getRX(), 
		basicMenu.getRY(), 
		basicMenu.getRS()
	  );
	}
	for(var i = 1; i < numElements+1; i++) {
	  var s = basicMenu.getES(i);
	  if (s > 0){
		col = newColors.searchNodeAt(i);
		c = col.data;
		fill(c);
		ellipse(
		  basicMenu.getEX(i), 
		  basicMenu.getEY(i), 
		  basicMenu.getES(i)
		);
	  }

	  if (basicMenu.selActive(i)){		//Click selection to apply
	  //if(basicMenu.getSel() == i){	//Apply selection automatically (3+ elements only)
		backgroundColor = c;
	  }
	}
  }
  else if ((!basicMenu.isOpen())) {
	backgroundColor = color(0,0,128);
  }
}

function touchStarted() {

}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  setup();
}

function keyPressed() {
  switch(keyCode) {
  case LEFT_ARROW:
	basicMenu.setX(basicMenu.getX() - 10);
    break;

  case RIGHT_ARROW:
	basicMenu.setX(basicMenu.getX() + 10);
    break;

  case UP_ARROW:
	basicMenu.setY(basicMenu.getY() - 10);
    break;

  case DOWN_ARROW:
	basicMenu.setY(basicMenu.getY() + 10);
    break;

  case 98:   //Numpad 2 (down)
    break;

  case 104:  //Numpad 8 (up)
    break;

  case 100:  //Numpad 4 (left)
    break;

  case 102:  //Numpad 6 (right)
    break;
  }

  switch(key) {
  case ' ':
    break;
  }
}
