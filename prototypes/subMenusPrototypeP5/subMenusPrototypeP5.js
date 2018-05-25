var cx;
var cy;
var parentMenu;
var subMenuR;
var subMenuG;
var subMenuB;
var backgroundColor;
var frameLimit = 24;
var numElements = 3;
var someReds;
var someGreens;
var someBlues;
var menuElements;

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

  parentMenu = new dropMenu(
	cx,						//x-coordinate of menu
	cy/2,					//y-coordinate of menu
	1.0,					//size of menu (1.0)
	PI,						//what degree (and direction) menu icon rotates when opening
	dBias,					//for dpi scaling
	frameLimit,						//duration, in frames, of animations. 0=animations disabled
	numElements,			//number of elements (length of list) for the menu to handle
	true,					//this menu must be clicked to open
	'RIGHT',				//this menu opens upwards
	);

  subMenuR = new dropMenu(
	cx,						//x-coordinate of menu
	cy/2,					//y-coordinate of menu
	1.0,					//size of menu (1.0)
	0,						//what degree (and direction) menu icon rotates when opening
	dBias,					//for dpi scaling
	frameLimit,						//duration, in frames, of animations. 0=animations disabled
	numElements,			//number of elements (length of list) for the menu to handle
	false,					//this menu must be clicked to open
	'DOWN',					//this menu opens upwards
	);
  subMenuG = new dropMenu(
	cx,						//x-coordinate of menu
	cy/2,					//y-coordinate of menu
	1.0,					//size of menu (1.0)
	0,						//what degree (and direction) menu icon rotates when opening
	dBias,					//for dpi scaling
	frameLimit,						//duration, in frames, of animations. 0=animations disabled
	numElements,			//number of elements (length of list) for the menu to handle
	false,					//this menu must be clicked to open
	'DOWN',					//this menu opens upwards
	);
  subMenuB = new dropMenu(
	cx,						//x-coordinate of menu
	cy/2,					//y-coordinate of menu
	1.0,					//size of menu (1.0)
	0,						//what degree (and direction) menu icon rotates when opening
	dBias,					//for dpi scaling
	frameLimit,						//duration, in frames, of animations. 0=animations disabled
	numElements,			//number of elements (length of list) for the menu to handle
	false,					//this menu must be clicked to open
	'DOWN',					//this menu opens upwards
	);

  backgroundColor = color(0,0,128);
  someReds 		= new SinglyList();
  someBlues		= new SinglyList();
  someGreens	= new SinglyList();
  for (var i = 0; i < numElements; i++){
	someReds.add  (color(60, (i+1)*(256/numElements), 128+i*64));
	someGreens.add(color(170, (i+1)*(256/numElements), 128+i*64));
	someBlues.add (color(255, (i+1)*(256/numElements), 128+i*64));
  }

  menuElements	= new SinglyList();
  menuElements.add("red");
  menuElements.add("green");
  menuElements.add("blue");
}

function draw() {
  background(backgroundColor);
  parentMenu.drawMenu();
  var selection = parentMenu.getSel();
  fill(backgroundColor);
  ellipse(
	parentMenu.getX(),
	parentMenu.getY(),
	dBias*0.30);
  if(parentMenu.isOpen()) {
/*
	var rs = parentMenu.getRS();
	if (rs > 0){
	  var rc = parentMenu.getR();
	  ellipse(
		parentMenu.getRX(), 
		parentMenu.getRY(), 
		parentMenu.getRS()
	  );
	}
*/
	if (parentMenu.getSel() == 1) {
	  subMenuR.setX(parentMenu.getEX(selection));
	  subMenuR.setY(parentMenu.getEY(selection));
	  subMenuR.setS(parentMenu.getES(selection)/dBias*4);
	  subMenuG.setX(parentMenu.getEX(selection+1));
	  subMenuG.setY(parentMenu.getEY(selection+1));
	  subMenuG.setS(parentMenu.getES(selection+1)/dBias*4);
	  subMenuB.setX(parentMenu.getEX(selection+2));
	  subMenuB.setY(parentMenu.getEY(selection+2));
	  subMenuB.setS(parentMenu.getES(selection+2)/dBias*4);
	  subMenuR.setOpen(true);
	  subMenuG.setOpen(false);
	  subMenuB.setOpen(false);
	  subMenuR.drawMenu();
	  subMenuG.drawMenu();
	  subMenuB.drawMenu();
	  col	= someReds.searchNodeAt(subMenuR.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuR.getX(),
		subMenuR.getY(),
		parentMenu.getES(selection));

	  col	= someGreens.searchNodeAt(subMenuG.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuG.getX(),
		subMenuG.getY(),
		parentMenu.getES(selection+1));

	  col	= someBlues.searchNodeAt(subMenuB.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuB.getX(),
		subMenuB.getY(),
		parentMenu.getES(selection+2));
  
	} 
	if (parentMenu.getSel() == 2) {
	  subMenuR.setX(parentMenu.getEX(selection-1));
	  subMenuR.setY(parentMenu.getEY(selection-1));
	  subMenuR.setS(parentMenu.getES(selection-1)/dBias*4);
	  subMenuG.setX(parentMenu.getEX(selection));
	  subMenuG.setY(parentMenu.getEY(selection));
	  subMenuG.setS(parentMenu.getES(selection)/dBias*4);
	  subMenuB.setX(parentMenu.getEX(selection+1));
	  subMenuB.setY(parentMenu.getEY(selection+1));
	  subMenuB.setS(parentMenu.getES(selection+1)/dBias*4);
	  subMenuR.setOpen(false);
	  subMenuG.setOpen(true);
	  subMenuB.setOpen(false);
	  subMenuR.drawMenu();
	  subMenuG.drawMenu();
	  subMenuB.drawMenu();
	  col	= someReds.searchNodeAt(subMenuR.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuR.getX(),
		subMenuR.getY(),
		parentMenu.getES(selection-1));
	  col	= someGreens.searchNodeAt(subMenuG.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuG.getX(),
		subMenuG.getY(),
		parentMenu.getES(selection));

	  col	= someBlues.searchNodeAt(subMenuB.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuB.getX(),
		subMenuB.getY(),
		parentMenu.getES(selection+1));
  
	}
	if (parentMenu.getSel() == 3) {
	  subMenuR.setX(parentMenu.getEX(selection-2));
	  subMenuR.setY(parentMenu.getEY(selection-2));
	  subMenuR.setS(parentMenu.getES(selection-2)/dBias*4);
	  subMenuG.setX(parentMenu.getEX(selection-1));
	  subMenuG.setY(parentMenu.getEY(selection-1));
	  subMenuG.setS(parentMenu.getES(selection-1)/dBias*4);
	  subMenuB.setX(parentMenu.getEX(selection));
	  subMenuB.setY(parentMenu.getEY(selection));
	  subMenuB.setS(parentMenu.getES(selection)/dBias*4);
	  subMenuR.setOpen(false);
	  subMenuG.setOpen(false);
	  subMenuB.setOpen(true);
	  subMenuR.drawMenu();
	  subMenuG.drawMenu();
	  subMenuB.drawMenu();
	  col	= someReds.searchNodeAt(subMenuR.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuR.getX(),
		subMenuR.getY(),
		parentMenu.getES(selection-2));
	  col	= someGreens.searchNodeAt(subMenuG.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuG.getX(),
		subMenuG.getY(),
		parentMenu.getES(selection-1));

	  col	= someBlues.searchNodeAt(subMenuB.getSel());
	  c		= color(col.data);  
	  fill(c);
	  ellipse(
		subMenuB.getX(),
		subMenuB.getY(),
		parentMenu.getES(selection));
  
	}
	if (parentMenu.getR() == 1) {
	  subMenuR.setX(parentMenu.getRX());
	  subMenuR.setY(parentMenu.getRY());
	  subMenuR.setS(parentMenu.getRS()/dBias*4);
	  subMenuR.drawMenu();
	} 
	if (parentMenu.getR() == 2) {
	  subMenuG.setX(parentMenu.getRX());
	  subMenuG.setY(parentMenu.getRY());
	  subMenuG.setS(parentMenu.getRS()/dBias*4);
	  subMenuG.drawMenu();
	}
	if (parentMenu.getR() == 3) {
	  subMenuB.setX(parentMenu.getRX());
	  subMenuB.setY(parentMenu.getRY());
	  subMenuB.setS(parentMenu.getRS()/dBias*4);
	  subMenuB.drawMenu();
	}
  }
  else if (!parentMenu.isOpen() && (parentMenu.getC() > 0)) {
	if (parentMenu.getSel() == 1) {
	  subMenuR.setX(parentMenu.getEX(selection));
	  subMenuR.setY(parentMenu.getEY(selection));
	  subMenuG.setX(parentMenu.getEX(selection+1));
	  subMenuG.setY(parentMenu.getEY(selection+1));
	  subMenuB.setX(parentMenu.getEX(selection+2));
	  subMenuB.setY(parentMenu.getEY(selection+2));
	} 
	if (parentMenu.getSel() == 2) {
	  subMenuR.setX(parentMenu.getEX(selection-1));
	  subMenuR.setY(parentMenu.getEY(selection-1));
	  subMenuG.setX(parentMenu.getEX(selection));
	  subMenuG.setY(parentMenu.getEY(selection));
	  subMenuB.setX(parentMenu.getEX(selection+1));
	  subMenuB.setY(parentMenu.getEY(selection+1));
	}
	if (parentMenu.getSel() == 3) {
	  subMenuR.setX(parentMenu.getEX(selection-2));
	  subMenuR.setY(parentMenu.getEY(selection-2));
	  subMenuG.setX(parentMenu.getEX(selection-1));
	  subMenuG.setY(parentMenu.getEY(selection-1));
	  subMenuB.setX(parentMenu.getEX(selection));
	  subMenuB.setY(parentMenu.getEY(selection));
	}
	subMenuR.drawMenu();
	subMenuG.drawMenu();
	subMenuB.drawMenu();
	subMenuR.setOpen(false);
	subMenuG.setOpen(false);
	subMenuB.setOpen(false);
	//backgroundColor = color(0,0,128);
  }
  var col;
  var c;
  for(var i = 1; (i < numElements + 1) && 
	parentMenu.getC() == frameLimit; i++){
	if (subMenuR.getES(i) > 0) {
	  col 	= someReds.searchNodeAt(i);
	  c		= color(col.data);
	  fill(c);
	  ellipse(
		subMenuR.getEX(i),
		subMenuR.getEY(i),
		subMenuR.getES(i));
	}	  
	if (subMenuR.selActive(i))
	  backgroundColor = c;
	if (subMenuG.getES(i) > 0) {
	  col 	= someGreens.searchNodeAt(i);
	  c		= color(col.data);
	  fill(c);
	  ellipse(
		subMenuG.getEX(i),
		subMenuG.getEY(i),
		subMenuG.getES(i));
	}	  
	if (subMenuG.selActive(i))
	  backgroundColor = c;
	if (subMenuB.getES(i) > 0) {
	  col 	= someBlues.searchNodeAt(i);
	  c		= color(col.data);
	  fill(c);
	  ellipse(
		subMenuB.getEX(i),
		subMenuB.getEY(i),
		subMenuB.getES(i));
	}	  
	if (subMenuB.selActive(i))
	  backgroundColor = c;
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
	parentMenu.setX(parentMenu.getX() - 10);
    break;

  case RIGHT_ARROW:
	parentMenu.setX(parentMenu.getX() + 10);
    break;

  case UP_ARROW:
	parentMenu.setY(parentMenu.getY() - 10);
    break;

  case DOWN_ARROW:
	parentMenu.setY(parentMenu.getY() + 10);
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
