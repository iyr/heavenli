var cx;
var cy;
var basicMenu;
var backgroundColor;

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
	cx,
	height-cy/2,
	1.0,
	PI,
	width,
	height,
	dBias,
	65,
	3,
	true,
	'UP',
	);
  backgroundColor = color(0,0,128);
}

function draw() {
  if((basicMenu.isOpen())) {
	for (i = 0; i < 3; i++){
	  if (basicMenu.selActive(i+1)){
		backgroundColor = color (i*64, 128, 128);
	  }
	}
  }
  else if ((!basicMenu.isOpen())) {
	backgroundColor = color(0,0,128);
  }

  //console.log((basicMenu.isOpen()));
  console.log((basicMenu.getSel()));
  background(backgroundColor);
/*
  if (basicMenu.selActive(1))
	background(255);
  else if (basicMenu.selActive(2))
	background(0);
  else
	background(128);
*/
  basicMenu.drawMenu();
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
