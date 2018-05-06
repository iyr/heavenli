/* HeavenLi Light Interface source code
 * For porting purposes, all 'vars' are floats unless otherwise noted.
 */

var interaction			= false;	//bool, 	coupled "isScreenTouched" + "interaction"
var frameLimit			= 65; 		//int,  	Length, in frames, of animations
var dBias				= 0;		//int,  	Display bias, the smaller of the screen's dimensions divided by two.
var timeSettingCursor	= 0;		//int,  	animation cursor for home screen / time setting screen transition
var colrSettingCursor	= 0;		//int,		animation cursor for home screen / color setting screen transition
var targetScreen		= 0;		//int,  	0: Home Screen
									//1: Color Setting Screen
									//2: Time Setting Screen
var targetBulb;						//int,  	which bulb the picker screen is adjusting
var targetClock;					//int,  	distinguishes time setting screen for time / alarm
var currentHue;						//int,  	currently selected hue
var currentBri;						//int,		currently selected brightness
var currentSat;						//int,		currently selected saturation
var prevHue;						//int,		previously set hue before color setting screen
var prevBri;						//int,		previously set brightness before color setting screen
var prevSat;						//int,		previously set saturation before color setting screen
var wereColorsTouched	= false;	//bool, 	Used for decoupling profile/bulb changes
var mode				= 1;		//int (for now), lighting arrangement: 1 = circular, -1 = linear

var angB				= 90;		//float,	angular offset for the background
var cx;								//float,	x-coord of the display's center
var cy;								//float,	y-coord of the display's center
var mx;								//float,	x-coord of the menu button (relative)
var my;								//float,	y-coord of the menu button (relative)
var mc 					= 0;		//int,		animation cursor for the menu button

var fx;								//float,	x-coord of the favorites button (relative)
var fy;								//float,	y-coord of the favorites button (relative)
var fc 					= 0;		//int,		animation cursor for the favorites button
var fsdc				= 0;		//int,		animation cursor for the save/delete button
var pc					= 0;		//int,		animation cursor for the profile selection menu
var pt 					= 1;		//int,		currently selected color profile
var pd					= 0;		//int,		scroll direction

var timeSetMode			= 0;		//int,		determines what mode (time/alarm) the time-setting screen is adjusting
var tHr					= 0;		//int,		current hour	(0-23)
var tMn					= 0;		//int,		current minute	(0-59)
var aHr					= 0;		//int, 		alarm hour		(0-23)
var aMn					= 0;		//int, 		alarm minute	(0-59)
var pHr					= 0;		//int,		"previous" variable for tracking changes		
var pMn					= 0;		//int,		"previous" variable for tracking changes		
var pta					= 0;		//int,		"previous" variable for tracking changes		
var meridiem;						//int,		1: PM, -1: AM
var oldcols = [];
var alarmDays = [];

var diff				= 1;		//int,		stores variable changes with respect to framerate
var numLights 			= 3;		//int,		number of discrete lights
var bulbsCurrentHSB		= [];		//array of ints that store the current values for each lamp
var bulbsTargetHSB		= [];		//array of ints that store the target values for animating light changes.
var savedFavList;

var touchHold 			= false;	//bool,		used for debouncing screen interactions
var lightOn 			= true;		//bool,		indicates if all lights are set to white light (gamma corrected)
var menuOpen 			= false;	//bool,		is the menu open
var favsOpen 			= false;	//bool,		^>favs

function setup() {
  colorMode(HSB, 255);
  frameRate(60);
  //pixelDensity(0.75);
  //createCanvas(dispWidth, dispHeight);
  createCanvas(windowWidth, windowHeight);
  background(0);
  noStroke();
  cx 		= width/2;
  cy 		= height/2;
  currentHue= 0;
  currentBri= 5;
  currentSat= 5;
  strokeCap(ROUND);
  strokeJoin(ROUND);
  meridiem = hour() > 12 ? 1 : -1;		

  //Configurations for portrait / landscape / square displays
  if (height > width) {
    dBias 	= cx;
    mx 		= width  - cx/4;
    my		= cy/5;
    fx		= width  - cx/4;
    fy		= height - cy/5;
  } else if (width > height) {
    dBias 	= cy;
    mx		= cx/5;
    my 		= height - cy/3;
    fx		= width  - cx/5;
    fy		= height - cy/3;
  } else if (width == height) {
    dBias	= cx;
    mx		= cx/4;
    my		= height - cy/4;
    fx		= width  - cx/4;
    fy		= height - cy/4;
  }

  for (var i = 0; i < 7; i++) {
	if (i > 0)
	  alarmDays[i] = false;
	else
	  alarmDays[i] = true;
  }
  // REIMPLEMENT THIS, PREFERABLY WITH LINKED LISTS
  for (var i = 0; i < 6; i++) {
    bulbsCurrentHSB[i] = color(0, 255, 0);
    bulbsTargetHSB[i] = color(255, 0, 255);
  }
  savedFavList = new SinglyList();
  preloadSampleFavorites(3);
}

function draw() {
  if (touches.length == 0 && !isMousePressed)
	interaction = false;
  if (touches.length >= 1 || isMousePressed)
	interaction = true;
  diff 	= constrain(60/frameRate(), 1, 2.4);

  /* Draw Home Screen */
  switch (targetScreen) {
  case 0:
    if (timeSettingCursor > 0) {
      timeSettingCursor = constrain(timeSettingCursor-3-diff, 0, frameLimit);
      drawSettingTime(timeSettingCursor, targetClock);
    }
    if (colrSettingCursor > 0) {
      colrSettingCursor = constrain(colrSettingCursor-3-diff, 0, frameLimit);
      drawSettingColor(colrSettingCursor, targetBulb);
    } 
    if (targetScreen == 0 &&
      timeSettingCursor == 0 &&
      colrSettingCursor == 0)
      drawHome();
    break;

  case 1:
    if (colrSettingCursor < frameLimit)
      colrSettingCursor = constrain(colrSettingCursor + 3 + diff, 0, frameLimit);
    drawSettingColor(colrSettingCursor, targetBulb);
    break;

  case 2:
    if (timeSettingCursor < frameLimit)
      timeSettingCursor = constrain(timeSettingCursor + 3 + diff, 0, frameLimit);
	  drawSettingTime(timeSettingCursor, targetClock);
    break;
  }

  updateColorZones();
  touchHold = interaction;
}

function touchStarted(){
  //necessarily empty
}
function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
  setup();
}

// Returns the angle between two points with the first point as the reference
function getAngFromPoints (jx, jy, kx, ky){
  if (jx <= kx && jy >= ky)
	return -degrees(atan((ky-jy)/(kx-jx)));

  if (jx >= kx && jy >= ky)
	return 180 - degrees(atan((ky-jy)/(kx-jx)));

  if (jx >= kx && jy <= ky)
	return 180 - degrees(atan((ky-jy)/(kx-jx)));

  if (jx <= kx && jy <= ky)
	return 360 - degrees(atan((ky-jy)/(kx-jx)));
}
function drawMenu(omx) {
  var acim	= animCurve(frameLimit-mc);
  var acbim = animCurveBounce(frameLimit-mc);
  var tm08	= dBias*0.08;
  var tm10	= dBias*0.1;
  var tm12	= dBias*0.12;
  var tm15	= dBias*0.15;
  var tm20	= dBias*0.2;
  var tm25	= dBias*0.25;
  var tm28	= dBias*0.28;
  var tm30	= dBias*0.3;
  var tm40	= dBias*0.4;
  strokeWeight(1);
  stroke(96);
  fill(240);

  noStroke();
  ellipse(mx+omx, my+((dBias-my)*0.525*acim), tm40, tm40);
  ellipse(mx+omx, my, tm40, tm40);
  //noStroke();
  rect(mx+omx-tm20, my, tm40, ((dBias-my)*.525*acim));
/*
  ellipse(mx+omx, my+((dBias-my)*1.05*acim), tm40, tm40);
  ellipse(mx+omx, my, tm40, tm40);
  //noStroke();
  rect(mx+omx-tm20, my, tm40, ((dBias-my)*1.05*acim));
*/
  stroke(96);
  //line(mx+omx-tm20-0.5, my, mx+omx-tm20-0.5, my+((dBias-my)*1.05*acim));
  //line(mx+omx+tm20-0.5, my, mx+omx+tm20-0.5, my+((dBias-my)*1.05*acim));

  strokeWeight(dBias*0.05);
  stroke(96);

  // Hamburger Iconography
  var xr	= tm08*cos(-animCurveBounce(mc)*HALF_PI);
  var yr	= tm08*sin(-animCurveBounce(mc)*HALF_PI);
  var tr	= tm08*(1-animCurveBounce(mc));
  line (mx+omx-yr, my-xr, mx+omx+yr, my+xr);
  line (mx+omx-yr-tr, my-xr+tr-tm08, mx+omx+yr-tr, my+xr+tr-tm08);
  line (mx+omx-yr+tr, my-xr-tr+tm08, mx+omx+yr+tr, my+xr-tr+tm08);
  noStroke();
  strokeWeight(1);

  if (mc > floor(frameLimit*0.25)) {
    fill(96+(255-96)*(1-acim));
/*
    // Draw mini clock
    //,mc < frameLimit/4 ? 0 : 255);
    ellipse(mx+omx, my+((dBias-my)*acim), tm28);
    stroke(255);
    line(mx+omx, my+((dBias-my)*acim), 
      mx+omx, my+((dBias-my)*(height > width ? 0.85 : 1.15)*acim));
      //mx+omx, my+((dBias-my)*(height > width ? 0.32 : 1-0.32)*acim));
    strokeWeight(2);
    line(mx+omx, my+((dBias-my)*acim), 
      mx+omx+tm08, my+((dBias-my)*acim));
*/

    // Draw alarm clock
    noStroke();
    ellipse(mx+omx, my+((dBias-my)*0.5*acim), tm28);
    //ellipse(mx+omx-tm08, my+((dBias-my)*(height > width ? 0.85 : 1.15)*0.5*acim), tm15);
    ellipse(mx+omx-tm08, my+((dBias-my)*(height > width ? 0.7 : 1+0.32)*0.5*acim), tm15);
    ellipse(mx+omx+tm08, my+((dBias-my)*(height > width ? 0.7 : 1+0.32)*0.5*acim), tm15);
/*
    triangle(mx+omx, 
      my+((dBias-my)*(height > width ? 1.1 : 0.9)*acim*0.5), 
      mx+omx-tm12, 
      my+((dBias-my)*(height > width ? 1.235 : 0.765)*acim*0.5), 
      mx+omx+tm12, 
      my+((dBias-my)*(height > width ? 1.235 : 0.765)*acim*0.5));
*/
    strokeWeight(1);
    stroke(255);
    line(mx+omx, my+((dBias-my)*acim*0.5), 
      mx+omx, my+((dBias-my)*(height > width ? 0.82 : 1.18)*acim*0.5));
    strokeWeight(2);
    line(mx+omx, my+((dBias-my)*acim*0.5), 
      mx+omx+tm08, my+((dBias-my)*acim*0.5));
  }

  // Watch Menu Button for input
  if (watchPoint(mx+omx, my, tm40, tm40) && !touchHold) {
	favsOpen = false;
    if (!menuOpen) {
      menuOpen = true;
    } else {
      menuOpen = false;
    }
    touchHold = true;
  } else if (!interaction) {
    touchHold = false;
  }

  // Watch clock button for input
  //if (watchPoint(mx+omx, my+((dBias-my)*acim), tm40))

  // Close Menu if open and somewhere else on screen is touched
  if (interaction &&
    menuOpen &&
    mc == frameLimit &&
	!touchHold
    ) {
	if (watchPoint(mx+omx, my+((dBias-my)*0.5*acim), tm40)){
	  targetScreen	= 2;
	  timeSetMode	= 1;
	  pHr			= tHr;
	  pMn			= tMn;
	  for (var i = 0; i < 6; i++) {
		oldcols[i]			= color(bulbsTargetHSB[i]);
		bulbsTargetHSB[i]	= color(0,0,0);
	  }
	}
	else /*
	if (watchPoint(mx+omx, my+((dBias-my)*acim), tm40)){
	  targetScreen	= 2;
	  timeSetMode	= 0;
	  pHr			= tHr;
	  pMn			= tMn;
	  for (var i = 0; i < numLights; i++) {
		oldcols[i]			= color(bulbsTargetHSB[i]);
		bulbsTargetHSB[i]	= color(0,0,0);
	  }
	}
*/
    menuOpen = false;
	}

  if (menuOpen)
    mc = constrain(mc + 6*diff, 0, frameLimit);
  else if (!menuOpen)
    mc = constrain(mc - 6*diff, 0, frameLimit);
}

function drawFavorites(ofx) {
  var acibf	= animCurveBounce(frameLimit-fc);
  var acif	= animCurve(frameLimit-fc);
  var acibp = animCurveBounce(frameLimit-pc);
  var het 	= height-dBias*1.33-fy;
  var tm04	= dBias*0.04;
  var tm06 	= dBias*0.06;
  var tm08  = dBias*0.08;
  var tm12	= dBias*0.12;
  var tm13 	= dBias*0.125;
  var tm15 	= dBias*0.15;
  var tm20	= dBias*0.2;
  var tm25	= dBias*0.25;
  var tm30	= dBias*0.3;
  var tm35	= dBias*0.35;
  var tm40	= dBias*0.4;
  var o;
  strokeWeight(1);
  stroke(96);
  fill(240);
  noStroke();
  ellipse(fx+ofx, fy, tm40, tm40);  

  switch(savedFavList.listSize()) {
  case 1:
    if (width > height)
      o = (dBias-fy)*0.5;
    if (width <= height)
      o = het*0.4;
    break;

  case 2:
    if (width > height)
      o = dBias-fy;
    if (width <= height)
      o = het*0.6;
    break;

  default:
    o = het;
  }

  ellipse(fx+ofx, fy+o*acif, tm40);
  //noStroke();
  rect(fx+ofx-tm20, fy, tm40, o*acif);
  stroke(96);
  noStroke();

  //Draw Heart Iconography
  fill(96);
  var xrs 	= cos(PI-acibf*PI);
  var yrs 	= sin(PI-acibf*PI);

  ellipse ( fx+ofx-tm06*xrs, 
    fy+tm06*yrs-tm04+tm08*acibf, 
    tm15);

  ellipse ( fx+ofx+tm06*xrs, 
    fy-tm06*yrs-tm04+tm08*acibf, 
    tm15);

  triangle (fx+ofx-tm13*xrs, 
    fy+tm15*yrs, 
    fx+ofx+tm13*xrs, 
    fy-tm15*yrs, 
    fx+ofx-tm13*yrs, 
    fy-tm15*xrs-tm04+tm08*acibf);

  fill(96+(255-96)*animCurve(fc), (fc >= floor(frameLimit*0.25) ? 255 : 0));

  // Watch Favorites Button for input
  if (watchPoint(fx+ofx, fy, tm40, tm40) && !touchHold) {
    if (!favsOpen) {
      favsOpen = true;
    } else {
      favsOpen = false;
    }
    touchHold = true;
  } else if (!interaction) {
    touchHold = false;
  }


  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 3+ profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  if (savedFavList.listSize() >=3) {
    var utri	= 1;
    var mtri	= 0.7;
    var ltri	= 0.4;
    ellipse(fx+ofx, fy+(o*acif), tm30);
    ellipse(fx+ofx, fy+(o*0.4*acif), tm30);
    rect (fx+ofx-tm15, 
      fy+(o*acif), 
      dBias*.3, 
      -(o*0.6*acif));

    fill(255, (fc >= floor(frameLimit*0.5) ? 255 : 0));
    ellipse(fx+ofx-tm04, fy+(o*mtri*acif), tm20);

    if (favsOpen && pc == frameLimit) {
      var profArgs = [];
      profArgs = [
        fx+ofx, 
        fy+(o*utri*acif), 
        fy+(o*mtri*acif), 
        fy+(o*ltri*acif), 
        pt >= savedFavList.listSize() ? 1 : pt + 1, 
        pt <= 1 ? savedFavList.listSize() : pt - 1
      ];

      if (mode > 0) {
        drawHomeCircle(profArgs[0], profArgs[1], tm13*0.67, tm13*0.67, numLights, angB, profArgs[4]);
        drawHomeCircle(profArgs[0], profArgs[2], tm13, tm13, 	   numLights, angB, pt);
        drawHomeCircle(profArgs[0], profArgs[3], tm13*0.67, tm13*0.67, numLights, angB, profArgs[5]);
      } 
      if (mode < 0) {
        drawHomeTri(profArgs[0], profArgs[1], tm13*0.67, tm13*0.67, numLights, angB, profArgs[4]);
        drawHomeTri(profArgs[0], profArgs[2], tm13, tm13, 	    numLights, angB, pt);
        drawHomeTri(profArgs[0], profArgs[3], tm13*0.67, tm13*0.67, numLights, angB, profArgs[5]);
      }
    } else if (favsOpen && pc != frameLimit && pd == -1) {
      //Scoll Up

      var profArgs = [];
      profArgs = [
        fx+ofx, 
        fy+(o*utri*acif)-het*0.25*(1-acibp), 
        fy+(o*mtri*acif)-het*0.25*(1-acibp), 
        fy+(o*ltri*acif), 
        pt >= savedFavList.listSize() ? 1 : pt + 1, 
        pt <= 1 ? savedFavList.listSize() : pt - 1, 
        tm13*(0.67+0.33*(1-acibp)), 
        tm13*(1-animCurveBounce(pc))
      ];
      if (mode > 0) {
        drawHomeCircle(profArgs[0], profArgs[1], profArgs[6], profArgs[6], numLights, angB, profArgs[4]);
        drawHomeCircle(profArgs[0], profArgs[2], tm13, 		  tm13, 		   numLights, angB, pt);
        drawHomeCircle(profArgs[0], profArgs[3], profArgs[7], profArgs[7], numLights, angB, profArgs[5]);
      } 
      if (mode < 0) {
        drawHomeTri(profArgs[0], profArgs[1], profArgs[6], profArgs[6], numLights, angB, profArgs[4]);
        drawHomeTri(profArgs[0], profArgs[2], tm13, 		   tm13, 	    numLights, angB, pt);
        drawHomeTri(profArgs[0], profArgs[3], profArgs[7], profArgs[7], numLights, angB, profArgs[5]);
      }
    } else if (favsOpen && pc != frameLimit && pd == 1) {
      //Scroll down
      var profArgs = [];
      profArgs = [
        fx+ofx, 									// 0
        fy+(o*utri*acif), 							// 1
        fy+(o*mtri*acif)+het*0.25*(1-acibp), 		// 2
        fy+(o*ltri*acif)+het*0.25*(1-acibp), 		// 3
        pt >= savedFavList.listSize() ? 1 : pt + 1, // 4
        pt <= 1 ? savedFavList.listSize() : pt - 1, // 5
        tm13*(1-animCurveBounce(pc)), 				// 6
        tm13*(0.67+0.33*acibp),						// 7
        tm13*(0.67+0.33*(1-acibp)) 					// 8
      ];
      if (mode > 0) {
        drawHomeCircle(profArgs[0], profArgs[1], profArgs[6], profArgs[6], numLights, angB, profArgs[4]);
        drawHomeCircle(profArgs[0], profArgs[2], profArgs[7], profArgs[7], numLights, angB, pt);
        drawHomeCircle(profArgs[0], profArgs[3], profArgs[8], profArgs[8], numLights, angB, profArgs[5]);
      } 
      if (mode < 0) {
        drawHomeTri(profArgs[0], profArgs[1], profArgs[6], profArgs[6], numLights, angB, profArgs[4]);
        drawHomeTri(profArgs[0], profArgs[2], profArgs[7], profArgs[7], numLights, angB, pt);
        drawHomeTri(profArgs[0], profArgs[3], profArgs[8], profArgs[8], numLights, angB, profArgs[5]);
      }
    }

    // Close Favorites if open and somewhere else on screen is touched
    if (interaction &&
      favsOpen &&
      fc == frameLimit &&
      !touchHold &&
      !watchPoint(fx+ofx, fy+(o*acif), tm35) &&
      !watchPoint(fx+ofx, fy+(o*0.4*acif), tm35) &&
      !watchPoint(fx+ofx, fy+(o*0.7*acif), tm35)
      ) {
      favsOpen = false;
    }

    if (watchPoint(fx+ofx, fy+(o*acif), tm35) &&
      fc == frameLimit &&
      !touchHold) {
      favsOpen = true;
      pt >= savedFavList.listSize() ? pt = 1 : pt += 1;
      touchHold = true;
      pc = 0;
      pd = 1;
    }

    if (watchPoint(fx+ofx, fy+(o*0.4*acif), tm35) &&
      fc == frameLimit &&
      !touchHold) {
      favsOpen = true;
      pt <= 1 ? pt = savedFavList.listSize() : pt -= 1;
      touchHold = true;
      pc = 0;
      pd = -1;
    }

    //Apply Selected profile if clicked
    if (fc == frameLimit &&
      watchPoint(fx+ofx, fy+(o*0.67*acif), tm35)) {
      lightOn = true;
      for (var i = 0; i < numLights; i++) {
        var c = savedFavList.searchNodeAt(pt);
        var cols = c.data;
        var f = color(cols[i]);
        bulbsTargetHSB[i] = f;
      }
    }
  } 
  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 2 profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  else if (savedFavList.listSize() == 2) {
    ellipse(fx+ofx, fy+(o*acif), tm30);
    ellipse(fx+ofx, fy+(o*0.5*acif), tm30);
    rect (fx+ofx-tm15, 
      fy+(o*0.50*acif), 
      tm30, 
      (o*0.50*acif));
    fill(255, (fc >= floor(frameLimit*0.5) ? 255 : 0));

	if (mode > 0) {
    drawHomeCircle(
      fx+ofx, 
      fy+o*acif, 
      tm13*0.8*acif, 
      tm13*0.8*acif, 
      numLights, 
      angB, 
      1
      );
    drawHomeCircle(
      fx+ofx, 
      fy+o*0.5*acif, 
      tm13*0.8*acif, 
      tm13*0.8*acif, 
      numLights, 
      angB, 
      2
      );
	} else if (mode < 0) {
    drawHomeTri(
      fx+ofx, 
      fy+o*acif, 
      tm13*0.8*acif, 
      tm13*0.8*acif, 
      numLights, 
      angB, 
      1
      );
    drawHomeTri(
      fx+ofx, 
      fy+o*0.5*acif, 
      tm13*0.8*acif, 
      tm13*0.8*acif, 
      numLights, 
      angB, 
      2
      );

	}

    // Close Favorites if open and somewhere else on screen is touched
    if (interaction &&
      favsOpen &&
      fc == frameLimit &&
      !touchHold &&
      !watchPoint(fx+ofx, fy+(o*acif), tm35) &&
      !watchPoint(fx+ofx, fy+(o*0.5*acif), tm35)
      ) {
      favsOpen = false;
    }

    //Apply Selected profile if clicked
    if (fc == frameLimit &&
      watchPoint(fx+ofx, fy+(o*acif), tm35)) {
      lightOn = true;
      for (var i = 0; i < numLights; i++) {
        var c 		= savedFavList.searchNodeAt(1);
        var cols 	= c.data;
        var f 		= color(cols[i]);
        bulbsTargetHSB[i] = f;
      }
    }

    if (fc == frameLimit &&
      watchPoint(fx+ofx, fy+(o*0.5*acif), tm35)) {
      lightOn = true;
      for (var i = 0; i < numLights; i++) {
        var c 		= savedFavList.searchNodeAt(2);
        var cols 	= c.data;
        var f 		= color(cols[i]);
        bulbsTargetHSB[i] = f;
      }
    }
  } 

  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 1 profile~~~~~~~~~~~~~~~~~~~~~~~~~ */
  else if (savedFavList.listSize() == 1) {
    ellipse(fx+ofx, fy+(o*acif), tm30);
    fill(255, (fc >= floor(frameLimit*0.5) ? 255 : 0));
	if (mode > 0)
    drawHomeCircle(
      fx+ofx, 
      fy+(o*acif), 
      tm13*0.85*acif, 
      tm13*0.85*acif, 
      numLights, 
      angB, 
      1 
      );
	if (mode < 0)
    drawHomeTri(
      fx+ofx, 
      fy+(o*acif), 
      tm13*0.85*acif, 
      tm13*0.85*acif, 
      numLights, 
      angB, 
      1 
      );

    // Close Favorites if open and somewhere else on screen is touched
    if (interaction &&
      favsOpen &&
      fc == frameLimit &&
      !touchHold &&
      !watchPoint(fx+ofx, fy+(o*acif), tm35)
      ) {
      favsOpen = false;
    }

    //Apply Selected profile if clicked
    if (fc == frameLimit &&
      watchPoint(fx+ofx, fy+(o*acif), tm35)) {
      lightOn = true;
      for (var i = 0; i < numLights; i++) {
        var c 		= savedFavList.searchNodeAt(1);
        var cols 	= c.data;
        var f 		= color(cols[i]);
        bulbsTargetHSB[i] = f;
      }
    }
  }

  if (pc != frameLimit) {
    pc = constrain(pc + 6*diff, 0, frameLimit);
  }
  if (favsOpen)
    fc = constrain(fc + 6*diff, 0, frameLimit);
  else if (!favsOpen) 
    fc = constrain(fc - 6*diff, 0, frameLimit);
}

function drawClock(
	cs, 	// float, clock size scalar
	hop, 	// int,	  clock-hand opacity
	bri		// float, clock brightness
	) {
  noStroke();
  fill(bri*96);
  ellipse(cx, cy, dBias*cs);
  stroke(255*hop);

  var m = targetScreen == 2 ? aMn : tMn;
  var h = targetScreen == 2 ? aHr : tHr;
  m = (targetScreen == 2 ? map(m, 0, 60, 0, TWO_PI) : map(minute(), 0, 60, 0, TWO_PI)) - HALF_PI;
  h = (targetScreen == 2 ? map(h, 0, 24, 0, TWO_PI*2) : map(hour(), 0, 24, 0, TWO_PI*2)) - HALF_PI;
  strokeWeight(dBias*0.02);
  line(cx, cy, cx + cos(m) * (dBias*0.4)*cs, cy + sin(m) * (dBias*0.4)*cs);
  strokeWeight(dBias*0.03);
  line(cx, cy, cx + cos(h) * (dBias*0.25)*cs, cy + sin(h) * (dBias*0.25)*cs);
  noStroke();

  // Watch clock for input
  if (watchPoint(cx, cy, dBias) && !touchHold && targetScreen == 0) {
    if (!lightOn) {
      lightOn = true;
      for (var i = 0; i < numLights; i++) {
        var h = 255;
        var s = 0;
        var b = 255;
        bulbsTargetHSB[i] = color(h, s, b);
      }
    } else {
      lightOn = false;
      for (var i = 0; i < numLights; i++)
        bulbsTargetHSB[i] = 0;
    }
    touchHold = true;
  } else if (!interaction) {
    touchHold = false;
  }
}

function animCurve(c) {
  return -2.25*pow(c/(frameLimit*1.5), 2) + 1;
}

function animCurveBounce(c) {
  if (c == frameLimit)
    return 0;
  else
    return -3*pow((c-(frameLimit*0.14167))/(frameLimit*1.5), 2)+1.02675926;
}

function watchPoint(px, py, pr) {
  var wasMousePressed = false;
  if (1 >= pow((mouseX-px), 2) / pow(pr/2, 2) + pow((mouseY - py), 2) / pow(pr/2, 2)) 
  {
	if (touches.length >= 1 || isMousePressed) {
	  //interaction		= true;
	  //wasMousePressed	= interaction;
	  wasMousePressed	= true;
	} else 
	if (touches.length == 0 && !isMousePressed) {
	  //interaction		= false;
	  //wasMousePressed	= interaction;
	  wasMousePressed	= false;
	}
  }
  return wasMousePressed;
}

function updateColorZones() {
  var colC;
  var colT;

  //Update Color Zones
  for (var i = 0; i < numLights; i++) {
    colC		= color(bulbsCurrentHSB[i]);
    colT		= color(bulbsTargetHSB[i]);
    if (colC != colT) {
      colorMode(RGB, 255);
      var curR	= red  (colC);
      var curG	= green(colC);
      var curB	= blue (colC);
      var tarR	= red  (colT);
      var tarG	= green(colT);
      var tarB	= blue (colT);
      var difR	= abs(curR - tarR);
      var difG	= abs(curG - tarG);
      var difB	= abs(curB - tarB);
      var rd = 0;
      var gd = 0;
      var bd = 0;

      if (difR > 3)
        tarR > curR	? rd = diff+i/numLights : rd = -(diff+i/numLights);
      if (difG > 3)
        tarG > curG ? gd = diff+i/numLights : gd = -(diff+i/numLights);
      if (difB > 3)
        tarB > curB ? bd = diff+i/numLights : bd = -(diff+i/numLights);

      colC = color(
        difR >= 3 ? curR + rd : tarR, 
        difG >= 3 ? curG + gd : tarG, 
        difB >= 3 ? curB + bd : tarB);

      bulbsCurrentHSB[i] = colC;
      colorMode(HSB, 255);
    }
  }
}

function preloadSampleFavorites(n) {
  if (n == 0)
    return;

  switch(n) {
  case 1:
    var favs = [];
    favs[0]	= color(0, 255, 255);
    favs[1]	= color(85, 255, 255);
    favs[2]	= color(171, 255, 255);
    favs[3]	= color(0, 255, 255);
    favs[4]	= color(85, 255, 255);
    favs[5]	= color(171, 255, 255);
    savedFavList.add(favs);
    break;

  case 2:
    preloadSampleFavorites(1);
    var favs = [];
    favs[0]	= color(128, 255, 255);
    favs[1]	= color(43, 255, 255);
    favs[2]	= color(213, 255, 255);
    favs[3]	= color(128, 255, 255);
    favs[4]	= color(43, 255, 255);
    favs[5]	= color(213, 255, 255);
    savedFavList.add(favs);
    break;

  case 3:
    preloadSampleFavorites(2);
    var favs = [];
    favs[0]	= color(0, 0, 255);
    favs[1]	= color(0, 0, 192);
    favs[2]	= color(0, 0, 128);
    favs[3]	= color(0, 0, 255);
    favs[4]	= color(0, 0, 192);
    favs[5]	= color(0, 0, 128);
    savedFavList.add(favs);
    break;

  default:
    for (var i = 0; i < n; i++) {
      var favs = []
        for (var j = 0; j < 6; j++)
        favs[j] = color(
          random(0, 8)*32, 
          random(0, 8)*32, 
          random(0, 8)*32);
      savedFavList.add(favs);
    }
    return;
  }
}

// This will probably get deprecated
function drawProfile(
  px, 	// float, X-coord
  py, 	// float, Y-coord
  pp, 	// int,   index of profile to draw
  ps, 	// float, size/scale
  pa		// int,	  alpha
  ) {
  pp	= constrain(pp, 1, savedFavList.listSize());
  var tm03	= dBias*0.03;
  var tm12	= dBias*0.12;
  var tm13	= dBias*0.125;
  var c;
  var prof 	= savedFavList.searchNodeAt(pp);
  var cols	= prof.data;
  c	= color(cols[1]);
  fill (hue(c), saturation(c), brightness(c), pa);
  triangle (px-(tm03)*ps, 
    py, 
    px-tm13*ps, 
    py+tm12*ps, 
    px-tm13*ps, 
    py-tm12*ps		);

  c 		= color(cols[0]);
  fill (hue(c), saturation(c), brightness(c), pa);
  triangle (px-(tm03)*ps, 
    py, 
    px+tm13*ps, 
    py, 
    px-tm13*ps, 
    py-tm12*ps		);

  c 		= color(cols[2]);
  fill (hue(c), saturation(c), brightness(c), pa);
  triangle (px-(tm03)*ps, 
    py, 
    px+tm13*ps, 			
    py, 
    px-tm13*ps, 
    py+tm12*ps		);
}

function drawZoneBulbButton(bx, by, br, bid) {
  noStroke();
  var tm02	= br*0.02;
  var tm10	= br*0.1;
  var tm13	= br*0.125;
  fill(96);
  ellipse(bx, by, br);
  fill(color(bulbsCurrentHSB[bid]));
  ellipse(bx, by-tm13, br/2);
  //ellipse(bx, by-tm13, br/2.5);
  stroke(255);
  strokeWeight(tm13/2);
  for (var i = 1; i < 4; i++)
    line (
	  bx-tm10, 
      by+tm02+tm10*i, 
      bx+tm10, 
      by-tm02+tm10*i);
	line (
	  bx-tm10,
	  by+tm02+tm10*3,
	  bx,
	  by+tm02+tm10*3.5
	  );
	line (
	  bx+tm10,
	  by-tm02+tm10*3,
	  bx,
	  by+tm02+tm10*3.5
	  );
}

function drawHome() {
  if (mode > 0)
    drawHomeCircle(cx, cy, cx, cy, numLights, angB, 0);
  else if (mode < 0)
    drawHomeTri(cx, cy, cx, cy, numLights, angB, 0);

  //Draw Bulb Buttons
  for (var i = 0; i < numLights; i++) {
    if (mode > 0) {
      var tbt = radians(i*(360/numLights)+(180/numLights)+angB)
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    } else if (mode < 0) {
      var tbt = radians(i*(180/constrain(numLights-1, 1, numLights))+(angB-angB%45));
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    }
    drawZoneBulbButton(tbx, tby, dBias*0.3, i); 
    if (watchPoint(tbx, tby, dBias*0.3) && 
      mc == 0 && 
      fc == 0 &&
      !touchHold) {
      targetScreen		= 1;
      targetBulb		= i;
      var preCol		= color(bulbsTargetHSB[i]);
      prevHue			= hue(preCol);
      prevBri			= brightness(preCol);
      prevSat			= saturation(preCol);
    }
  }

  //Draw Clock
  drawClock(1, 1, 1);

  //Draw Menu Button
  drawMenu(0);

  //Draw Favorites button if there are any saved user profiles
  if (0 < savedFavList.listSize())
    drawFavorites(0);
}

/* Checks whether the targetBulb Colors match any existing profile
 * Returns 0 if the color configuration does not exist
 * Returns the index of the profile in the favorites list if it exists
 */
function isProfile() {
  var matches = 0;
  var prof;
  var cols;
  var c;
  var cT;
  for (var i = 1; i < savedFavList.listSize()+1; i++) {
    matches = 0;
    prof	= savedFavList.searchNodeAt(i);
    cols	= prof.data;
    for (var j = 0; j < numLights; j++) {
      c		= color(cols[j]);
      cT	= color(bulbsTargetHSB[j]);
      if (	   hue(c) == hue(cT) 		&&
        saturation(c) == saturation(cT) &&
        brightness(c) == brightness(cT)) {
        matches += 1;
      }
    }
    if (matches >= numLights) {
      return i;
    }
  }
  colorMode(HSB, 255);
  return 0;
}

function drawSettingColor(cursor, tb) {
  var acbic	= animCurveBounce(frameLimit-cursor);
  var acic	= animCurve(frameLimit-cursor);
  var acbc	= animCurveBounce(cursor);
  var cmx	= (width >= height ? mx : cx/4);
  var tm04	= dBias*0.04;
  var tm05	= dBias*0.05;
  var tm06	= dBias*0.06;
  var tm08	= dBias*0.08;
  var tm10	= dBias*0.1;
  var tm13	= dBias*0.13;
  var tm15	= dBias*0.15;
  var tm17	= dBias*0.17;
  var tm20  = dBias*0.2;
  var tm23	= dBias*0.23;
  var tm37	= dBias*0.37;
  var tm30	= dBias*0.3;
  var tm40	= dBias*0.4;
  var tm50	= dBias*0.5;
  var tm70	= dBias*0.7;
  var isPro = isProfile();

  if (isPro > 0)
    fsdc	= constrain(fsdc+3+diff, 0, frameLimit); // fsdc = frameLimit if profile exists
  if (isPro <= 0)
    fsdc	= constrain(fsdc-3-diff, 0, frameLimit); // fsdc = 0 if profile does not exist

  if (mode > 0)
	drawHomeCircle(cx, cy, cx, cy, numLights, angB, 0);
  if (mode < 0)
	drawHomeTri(cx, cy, cx, cy, numLights, angB, 0);

  //Draw Buttons
  for (var i = 0; i < numLights; i++) {
    //var tbx = cx + cos(radians(i*(360/numLights)+(180/numLights)+angB))*dBias*0.75;
    //var tby = cy + sin(radians(i*(360/numLights)+(180/numLights)+angB))*dBias*0.75;
    if (mode > 0) {
      var tbt = radians(i*(360/numLights)+(180/numLights)+angB)
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    } else if (mode < 0) {
      var tbt = radians(i*(180/constrain(numLights-1, 1, numLights))+(angB-angB%45));
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    }
    drawZoneBulbButton(tbx, tby, tm30*(1-acic), i);
  }

  //Draw Clock
  drawClock(1+0.7*acbic, 1-acic, 1-acic);

  // Draw Ring of hue circles
  var c 	= bulbsCurrentHSB[tb];
  var tmr	= tm30*acbic;
  for (var i = 0; i < 12; i++) {
    var ang = i*(360/12)+90;
    var tmx = cx + cos(radians(ang))*dBias*0.67*acbic;
    var tmy = cy + sin(radians(ang))*dBias*0.67*acbic;
    if (i*21.25 == currentHue)
    {
      if (wereColorsTouched == true)
        fill(240);
      else
        fill(128);
      ellipse(tmx, tmy, tmr+tm05);
      stroke(0);
      strokeWeight(1);
    } else {
      noStroke();
    }
    fill(color(i*21.25, 255*acic, 255));
    ellipse(tmx, tmy, tmr);
    if (watchPoint(tmx, tmy, tmr) && !touchHold) {
      wereColorsTouched = true;
      currentHue = i*21.25;
      bulbsTargetHSB[tb] = color(currentHue, currentSat, (255-currentBri));
    }
  }

  // Draw Saturation/Brightness triangle
  tmr = tm10*acbic;
  for (var i = 0; i < 6; i++) {
    for (var j = 0; j < 6-i; j++)
    {
      var tmx = cx-tm23+(i*tm13);
      var tmy = cy-tm37+(i*dBias*0.075+j*dBias*0.15);
      if (j*42.667 == currentBri && 
        i*42.667 == currentSat) {
        if (wereColorsTouched == true)
          fill(240);
        else
          fill(128);
        ellipse(tmx, tmy, tmr+tm05);
        stroke(0);
        strokeWeight(1);
      } else {
        noStroke();
      }
      fill(currentHue, i*42.667, 255-j*42.667);
      ellipse(tmx, tmy, 	tmr);
      if (watchPoint(tmx, tmy, tmr+tm05)) {
        wereColorsTouched	= true
          bulbsTargetHSB[tb]	= color(currentHue, i*42.667, 255-j*42.667);
        currentBri			= j*42.667;
        currentSat			= i*42.667;
      }
    }
  }

  //Draw Buttons
  strokeWeight(1);
  fill(240);
  noStroke();

  // Confirm
  ellipse(fx, fy+dBias*acbc, tm40);  
  // Cancel
  ellipse(cmx, fy+dBias*acbc, tm40);  

  // Draw Save/Delete Profile Button
  ellipse(fx, tm23 - dBias*acbc, tm40);

  stroke(96);
  strokeWeight(tm05);

  // Draw Check mark for confirm button
  line (fx-tm10, fy+dBias*acbc, 
    fx, fy+dBias*acbc+tm10);
  line (fx, fy+dBias*acbc+tm10, 
    fx+tm10, fy+dBias*acbc-dBias*0.12);

  // Draw Back arrow for cancel button
  noFill();
  cmx -= dBias*0.04;
  ellipse(cmx+tm08, fy+dBias*acbc+dBias*0.025, tm15);
  noStroke();
  fill(240);
  if (height > width)
    rect(cmx-tm08, fy+dBias*acbc-dBias*0.025, cmx-dBias*0.06, tm15);
  else if (height <= width)
    rect(cmx-tm08, fy+dBias*acbc-dBias*0.025, tm15, tm15);
  stroke(96);
  line (cmx-tm05, fy+dBias*acbc+tm10, 
    cmx+tm08, fy+dBias*acbc+tm10);
  line (cmx-tm10, fy+dBias*acbc-tm05, 
    cmx+tm08, fy+dBias*acbc-tm05);
  line (cmx, fy+dBias*acbc-tm08-tm05, 
    cmx-tm10, fy+dBias*acbc-tm05);
  line (cmx-tm10, fy+dBias*acbc-tm05, 
    cmx, fy+dBias*acbc+tm08-tm05);
  cmx += dBias*0.04;

  noStroke();
  fill(96);
  // Draw Heart and +/x iconography
  ellipse (	fx-tm06, 		0.19*dBias-dBias*acbc, tm15);
  ellipse (	fx+tm06, 		0.19*dBias-dBias*acbc, tm15);
  triangle(	fx-0.12*dBias, 	0.235*dBias-dBias*acbc, 
    fx+0.12*dBias, 	0.235*dBias-dBias*acbc, 
    fx, 				0.35*dBias-dBias*acbc);
  stroke(240);
  strokeWeight(0.03*dBias);
  // Profile Does Not Exist
  if (fsdc == 0) { 	
    line (fx-tm05, 0.22*dBias-dBias*acbc, 
      fx+tm05, 0.22*dBias-dBias*acbc);
    line (fx, 0.22*dBias-dBias*acbc-tm05, 
      fx, 0.22*dBias-dBias*acbc+tm05);
  } else 
  if (fsdc == frameLimit) {  //Profile Exists
    stroke(0, 240, 255);
    line (fx-tm04, 0.22*dBias-dBias*acbc-tm04, 
      fx+tm04, 0.22*dBias-dBias*acbc+tm04);
    line (fx+tm04, 0.22*dBias-dBias*acbc-tm04, 
      fx-tm04, 0.22*dBias-dBias*acbc+tm04);
  } else if (fsdc != 0 && fsdc != frameLimit) {
    var xrs1	= cos(-QUARTER_PI-animCurveBounce(fsdc)*QUARTER_PI);
    var yrs1	= sin(-QUARTER_PI-animCurveBounce(fsdc)*QUARTER_PI);
    var xrs2	= cos(QUARTER_PI-animCurveBounce(fsdc)*QUARTER_PI);
    var yrs2	= sin(QUARTER_PI-animCurveBounce(fsdc)*QUARTER_PI);
    stroke(0, 240*animCurve(frameLimit-fsdc), 255);
    line (fx+tm05*xrs1, 0.22*dBias-dBias*acbc-tm05*yrs1, 
      fx-tm05*xrs1, 0.22*dBias-dBias*acbc+tm05*yrs1);
    line (fx-tm05*xrs2, 0.22*dBias-dBias*acbc+tm05*yrs2, 
      fx+tm05*xrs2, 0.22*dBias-dBias*acbc-tm05*yrs2);
  }

  //Watch Confirm button for input
  if (watchPoint(fx, fy+dBias*acbc, tm40) && !touchHold)
  {
    if (wereColorsTouched) {
      bulbsTargetHSB[tb] = color(currentHue, currentSat, 255-currentBri);
      lightOn = true;
    }
    wereColorsTouched = false;
    targetScreen = 0;
  }

  //Watch Cancel Button for input
  if (watchPoint(cmx, fy+dBias*acbc, tm40) && !touchHold)
  {
    bulbsTargetHSB[tb] = color(prevHue, prevSat, prevBri);
    wereColorsTouched = false;
    targetScreen = 0;
  }

  // Watch Save/Delete Button for input
  if (watchPoint(fx, tm23-dBias*acbc, tm40) && !touchHold)
  {
    if (fsdc == 0) {
      var favs		= [];
      for (var i = 0; i < 6; i++) {
        try {
          favs[i]		= color(bulbsTargetHSB[i]);
        }
        catch(err) {
          favs[i]		= color(0, 0, i*60);
        }
      }
      savedFavList.add(favs);
    } else if (fsdc == frameLimit) {
      savedFavList.remove(isPro);
    }
  }

  //Draw Menu Button (for animation)
  if (height > width)
    drawMenu(dBias*acbic);
  else if (height <= width)
    drawMenu(-dBias*acbic);

  //Draw Favorites button if there are any saved user profiles
  if (0 < savedFavList.listSize())
    drawFavorites(dBias*acbic);
}


function drawSettingTime(cursor, tc, cMode) {
  var mnNew	= aMn*6-90;
  var hrNew	= aHr*30-90;
  var acbic	= animCurveBounce(frameLimit-cursor);
  var acic	= animCurve(frameLimit-cursor);
  var acbc	= animCurveBounce(cursor);
  var cmx	= (width >= height ? mx : cx/4);
  var tm05	= dBias*0.05;
  var tm08	= dBias*0.08;
  var tm10	= dBias*0.10;
  var tm15	= dBias*0.15;
  var tm20	= dBias*0.20;
  var tm25	= dBias*0.25;
  var tm30	= dBias*0.30;
  var tm35	= dBias*0.35;
  var tm40	= dBias*0.40;

  if (mode > 0)
	drawHomeCircle(cx, cy, cx, cy, numLights, angB, 0);
  if (mode < 0)
	drawHomeTri(cx, cy, cx, cy, numLights, angB, 0);

  //Draw Buttons
  for (var i = 0; i < numLights; i++) {
    if (mode > 0) {
      var tbt = radians(i*(360/numLights)+(180/numLights)+angB)
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    } else if (mode < 0) {
      var tbt = radians(i*(180/constrain(numLights-1, 1, numLights))+(angB-angB%45));
      var tbx = cx + cos(tbt)*dBias*0.75;
      var tby = cy + sin(tbt)*dBias*0.75;
    }
    drawZoneBulbButton(tbx, tby, tm30*(1-acic), i);
  }

  drawClock(1+0.5*acbic, 1, 1);

  //draw useful tickMarks
  stroke(225, 128*acbic);

  for (var i = 0; i < 60; i++) {
	if (i%5 == 0) {
	  strokeWeight(2);
	  line(
	    cx + cos(radians(i*(360/12)))*dBias*0.63,
	    cy + sin(radians(i*(360/12)))*dBias*0.63,
	    cx + cos(radians(i*(360/12)))*dBias*0.73,
	    cy + sin(radians(i*(360/12)))*dBias*0.73
	  ); }
	else {
	  strokeWeight(1);
	  line(
	    cx + cos(radians(i*6))*dBias*0.70,
	    cy + sin(radians(i*6))*dBias*0.70,
	    cx + cos(radians(i*6))*dBias*0.74,
	    cy + sin(radians(i*6))*dBias*0.74
	  );
	}
  }

  stroke(225, 128*acbic);
  noFill();
  ellipse(cx, cy, tm25*4);

  //noStroke();
  fill(255, 255*acbic)
  // Watch Clock hands for input
  if (
	watchPoint(cx, cy, tm40*4) && 
	!watchPoint(cx, cy, tm25*4) && 
	timeSettingCursor == frameLimit &&
	pta >= 0
	) {
	pta = 1;
    var tma	= 90-floor(getAngFromPoints(cx, cy, mouseX, mouseY));
	console.log(tma);
	mnNew = tma-90;
	aMn = mnNew/6-45;
  }

  if (
	watchPoint(cx, cy, tm25*4) && 
	timeSettingCursor == frameLimit &&
	pta <= 0
	) {
	pta = -1;
    var tma	= 90-floor(getAngFromPoints(cx, cy, mouseX, mouseY));
	console.log(tma);
	hrNew = (tma-90) - (tma-90)%30;
	aHr = hrNew/30-45;
  }

  if (!touchHold)
	pta = 0;

  ellipse(
	cx + cos(radians(mnNew))*tm40*(1+0.5*acbic),
	cy + sin(radians(mnNew))*tm40*(1+0.5*acbic),
	tm10,
	tm10);
  ellipse(
	cx + cos(radians(hrNew))*tm25*(1+0.5*acbic),
	cy + sin(radians(hrNew))*tm25*(1+0.5*acbic),
	tm10,
	tm10);
  //noFill();

  strokeWeight(1);

  // Confirm
  fill(240);
  noStroke();
  ellipse(fx, fy+dBias*acbc, tm40);  
  // Cancel
  fill(0);
  stroke(240);
  ellipse(cmx, fy+dBias*acbc, tm40);  

  // AM/PM slider
  var tmofy;
  switch(dBias) {
  case width/2:
	tmofy = tm08;
	break;
  case height/2:
	tmofy = tm20	
	break;
}
  if (
	watchPoint(
	  cx-tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20)
	)
	meridiem = -1;
  if (
	watchPoint(
	  cx+tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20)
	)
	meridiem = 1;
	
  if (meridiem < 0) {
	fill(240);
	noStroke();
	ellipse(
	  cx-tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20);

	fill(0);
	stroke(240);
	ellipse(
	  cx+tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20);
  } else {
	fill(240);
	noStroke();
	ellipse(
	  cx+tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20);

	fill(0);
	stroke(240);
	ellipse(
	  cx-tm10, 
	  fy+dBias*acbc+tmofy,
	  tm20);
  } 
  noStroke();
  textSize(tm15);
  fill(96);
  text('A',
	cx-tm15, 
	fy+dBias*acbc+tmofy+tm05
	);
  text('P',
	cx+tm05, 
	fy+dBias*acbc+tmofy+tm05
	);

  // Week Days
  var weekDays = ['S','M','T','w','T','F','S'];
  for (var i = 0; i < 7; i++) {
	strokeWeight(2);
	stroke(240);
	if (alarmDays[i])
	  fill(240);
	else
	  fill(0);
	ellipse(
	  cx-dBias*0.75+tm25*i, 
	  tm20*acbic-tm10, 
	  tm20);
	fill(96);
	if (!touchHold &&
	  watchPoint(
		cx-dBias*0.75+tm25*i, 
		tm20*acbic-tm10, 
		tm20)) {
	  alarmDays[i] = !alarmDays[i];
	}	
	
	noStroke();
	text(
	  weekDays[i], 
	  cx-dBias*0.80+tm25*i, 
	  tm20*acbic-tm05
	  );
  }

  // Draw Check mark for confirm button
  stroke(96);
  strokeWeight(tm05);
  line (fx-tm10, fy+dBias*acbc, 
    fx, fy+dBias*acbc+tm10);
  line (fx, fy+dBias*acbc+tm10, 
    fx+tm10, fy+dBias*acbc-dBias*0.12);

  // Draw Back arrow for cancel button
  stroke(240);
  strokeWeight(tm05);
  fill(0);
  cmx -= dBias*0.04;
  ellipse(cmx+tm08, fy+dBias*acbc+dBias*0.025, tm15);
  noStroke();
  fill(0);
  if (height > width)
    rect(cmx-tm08, fy+dBias*acbc-dBias*0.025, cmx-dBias*0.06, tm15);
  else if (height <= width)
    rect(cmx-tm08, fy+dBias*acbc-dBias*0.025, tm15, tm15);
  stroke(240);
  line (cmx-tm05, fy+dBias*acbc+tm10, 
    cmx+tm08, fy+dBias*acbc+tm10);
  line (cmx-tm10, fy+dBias*acbc-tm05, 
    cmx+tm08, fy+dBias*acbc-tm05);
  line (cmx, fy+dBias*acbc-tm08-tm05, 
    cmx-tm10, fy+dBias*acbc-tm05);
  line (cmx-tm10, fy+dBias*acbc-tm05, 
    cmx, fy+dBias*acbc+tm08-tm05);
  cmx += dBias*0.04;

  noStroke();
  fill(240);

  //Watch Confirm button for input
  if (watchPoint(fx, fy+dBias*acbc, tm40) && !touchHold)
  {
	for (var i = 0; i < 6; i++) {
	  var tmCol			= color(oldcols[i]);
	  var tmHue			= hue(tmCol);
	  var tmSat			= saturation(tmCol);
	  var tmBri			= brightness(tmCol);
	  bulbsTargetHSB[i] = color(tmHue, tmSat, tmBri);
	}
    targetScreen = 0;
  }

  //Watch Cancel Button for input
  if (watchPoint(cmx, fy+dBias*acbc, tm40) && !touchHold)
  {
	for (var i = 0; i < 6; i++) {
	  var tmCol			= color(oldcols[i]);
	  var tmHue			= hue(tmCol);
	  var tmSat			= saturation(tmCol);
	  var tmBri			= brightness(tmCol);
	  bulbsTargetHSB[i] = color(tmHue, tmSat, tmBri);
	}
    targetScreen 	= 0;
	aHr				= pHr;
	aMn				= pMn;
  }

  //Draw Menu Button (for animation)
  if (height > width)
    drawMenu(dBias*acbic);
  else if (height <= width)
    drawMenu(-dBias*acbic);

  //Draw Favorites button if there are any saved user profiles
  if (0 < savedFavList.listSize())
    drawFavorites(dBias*acbic);
}

function drawHomeCircle(
  gx, 		// float, X-coordinate of where the background (as a shape) is drawn
  gy, // float, Y-coordinate of where the background (as a shape) is drawn
  dx, 		// float, width (radius) of draw space
  dy, 		// float, height (radius) of draw space
  nz, 		// int,   number of zones to divide, correlates to number of bulbs
  ao, 		// float, angular offset of the colored zones
  colMode	// int,   0: uses colors of 'bulbsCurrentHSB', 1+: colors of indexed user-saved profile 
  ) {
  for (var i = 0; i < nz; i++) {
    // Green Lines
    var px = constrain(gx + cos(radians(i*(360/nz)+ao+45))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
    var py = constrain(gy + sin(radians(i*(360/nz)+ao+45))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);
    // Blue Lines
    var qx = constrain(gx + cos(radians(i*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
    var qy = constrain(gy + sin(radians(i*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);
    // Next Line in iteration for zoning
    var rx = constrain(gx + cos(radians((i+1)*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gx-dx, gx+dx);
    var ry = constrain(gy + sin(radians((i+1)*(360/nz)+ao))*sqrt(pow(dx, 2) + pow(dy, 2)), gy-dy, gy+dy);

    if (colMode <= 0) {
      fill(bulbsCurrentHSB[i]);
      stroke(bulbsCurrentHSB[i]);
    } else if (colMode > 0) {
      var c;
      var prof = savedFavList.searchNodeAt(colMode);
      var cols = prof.data;
      c	 = color(cols[i]);
      fill(hue(c), saturation(c), brightness(c));
      stroke(hue(c), saturation(c), brightness(c));
    }

    line(px, py, qx, qy);
    line(px, py, rx, ry);

    switch(nz) {
    case 1:
      //fill(color(bulbsCurrentHSB[0]));
      rect(gx-dx, gy-dy, 2*dx, 2*dy);
      break;

    case 2:
      if (qx == gx-dx && rx == gx+dx) {
        quad(rx, ry, qx, qy, gx-dx, gy-dy, gx+dx, gy-dy);
      }
      if (qy == gy-dy && ry == gy+dy) {
        quad(rx, ry, qx, qy, gx+dx, gy-dy, gx+dx, gy+dy);
      }
      if (qx == gx+dx && rx == gx-dx) {
        quad(qx, qy, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);
      }
      if (qy == gy+dy && ry == gy-dy) {
        quad(qx, qy, rx, ry, gx-dx, gy-dy, gx-dx, gy+dy);
      }
      break;

    default:
      /* General Purpose Background Geometry Rendering for Circularly Arranged Lights
       * As of writing, all edge cases (the oodles of conditionals) were accounted for
       * 1 - 6 lights 
       */
      if (px == qx || py == qy)
        triangle(px, py, qx, qy, 
          (nz == 3) && (colMode <= 0) ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
          (nz == 3) && (colMode <= 0) ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);
      if (qx == rx || qy == ry)
        triangle(rx, ry, qx, qy, 
          (nz == 3) && (colMode <= 0) ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
          (nz == 3) && (colMode <= 0) ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);
      if (rx == px || ry == py)
        triangle(px, py, rx, ry, 
          (nz == 3) && (colMode <= 0) ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
          (nz == 3) && (colMode <= 0) ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

      if (px != qx && py != qy)
      {
        triangle(px, py, qx, qy, 
          (nz == 3) && (colMode <= 0) ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
          (nz == 3) && (colMode <= 0) ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

        triangle(px, py, qx, qy, 
          (px == gx-dx) || (px == gx+dx) ? px : qx, 
          (py == gy-dy) || (py == gy+dy) ? py : qy
          );

        if (
          (px > gx-dx) && 
          (px < gx+dx) &&
          (qx > gx-dx) &&
          (qx < gx+dx)
          ) {
          if ((px > gx-dx) && (px < gx))
            quad(px, py, qx, qy, gx-dx, gy-dy, gx-dx, gy+dy);

          if ((px < gx+dx) && (px > gx))
            quad(px, py, qx, qy, gx+dx, gy+dy, gx+dx, gy-dy);

          if ((qx > gx-dx) && (qx < gx))
            quad(qx, qy, px, py, gx-dx, gy-dy, gx-dx, gy+dy);

          if ((qx < gx+dx) && (qx > gx))
            quad(qx, qy, px, py, gx+dx, gy+dy, gx+dx, gy-dy);
        }
        if (
          (py > gy-dy) && 
          (py < gy+dy) &&
          (qy > gy-dy) &&
          (qy < gy+dy)
          ) {
          if ((py > gy-dy) && (py < gy))
            quad(px, py, qx, qy, gx-dx, gy-dy, gx+dx, gy-dy);

          if ((py < gy+dy) && (py > gy))
            quad(px, py, qx, qy, gx+dx, gy+dy, gx-dx, gy+dy);

          if ((qy > gy-dy) && (qy < gy))
            quad(qx, qy, px, py, gx-dx, gy-dy, gx-dx, gy+dy);

          if ((qy < gy+dy) && (qx > gy))
            quad(qx, qy, px, py, gx+dx, gy+dy, gx+dx, gy-dy);
        }
      }
      if (rx != px && ry != py)
      {
        triangle(px, py, rx, ry, 
          (nz == 3) && (colMode <= 0) ? gx + ( cos(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gx, 
          (nz == 3) && (colMode <= 0) ? gy + (-sin(radians(ao*nz))*dBias/3)*((cos(radians(ao*nz*4))+1)/2) : gy);

        triangle(px, py, rx, ry, 
          (px == gx-dx) || (px == gx+dx) ? px : rx, 
          (py == gy-dy) || (py == gy+dy) ? py : ry
          );

        if (
          (px > gx-dx) && 
          (px < gx+dx) &&
          (rx > gx-dx) &&
          (rx < gx+dx)
          ) {
          if ((px > gx-dx) && (px < gx))
            quad(px, py, rx, ry, gx-dx, gy-dy, gx-dx, gy+dy);

          if ((px < gx+dx) && (px > gx))
            quad(px, py, rx, ry, gx+dx, gy+dy, gx+dx, gy-dy);

          if ((rx > gx-dx) && (rx < gx))
            quad(gx-dx, gy-dy, gx-dx, gy+dy, rx, ry, px, py);

          if ((rx < gx+dx) && (rx > gx))
            quad(gx+dx, gy+dy, gx+dx, gy-dy, rx, ry, px, py);
        }
        if (
          (py > gy-dy) && 
          (py < gy+dy) &&
          (ry > gy-dy) &&
          (ry < gy+dy)
          ) {
          if ((py > gy-dy) && (py < gy))
            quad(px, py, rx, ry, gx-dx, gy-dy, gx+dx, gy-dy);

          if ((py < gy+dy) && (py > gy))
            quad(px, py, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);

          if ((ry > gy-dy) && (ry < gy))
            quad(rx, ry, px, py, gx-dx, gy-dy, gx+dx, gy-dy);

          if ((ry < gy+dy) && (ry > gy))
            quad(px, py, rx, ry, gx-dx, gy+dy, gx+dx, gy+dy);
        }
      }

      break;
    }
  }
}

function keyPressed() {
  switch(keyCode) {
  case LEFT_ARROW:
    if (mode > 0)
      angB -= 5.0;
    if (mode < 0) 
      angB -= 45.0;
    if (angB < 0)
      angB += 360;

    //println(ao);
    break;

  case RIGHT_ARROW:
    if (mode > 0)
      angB += 5.0;
    if (mode < 0)
      angB += 45.0;
    if (angB > 360)
      angB -= 360;

    //println(ao);
    break;

  case UP_ARROW:
    if (mode > 0)
      numLights = constrain(numLights+1, 1, 6);
    if (mode < 0)
      numLights = constrain(numLights+1, 1, 5);
    break;

  case DOWN_ARROW:
    if (mode > 0)
      numLights = constrain(numLights-1, 1, 6);
    if (mode < 0)
      numLights = constrain(numLights-1, 1, 5);
    break;

  case 98:   //Numpad 2 (down)
    wy -= 5;
    break;

  case 104:  //Numpad 8 (up)
    wy += 5;
    break;

  case 100:  //Numpad 4 (left)
    wx -= 5;
    break;

  case 102:  //Numpad 6 (right)
    wx += 5;
    break;
  }

  switch(key) {
  case ' ':
    mode *= -1;
    angB = 0;
    if (mode < 0)
      numLights = constrain(numLights, 1, 5);
    break;
  }
}

function drawHomeTri (
  gx, 	// float, X-coord of where the shape is drawn
  gy, 	// float, Y-coord or where the shape is drawn
  dx, 		// float, width (radius) of the draw space
  dy, 		// float, height (radius) of the draw space
  nz, 		// int,	  number of zones to divide, correlates to number of bulbs
  ao, 		// float, angular offset of the colored zones
  colMode // int,	  0: uses colors of 'bulbsCurrentHSB', 1+: colors of indexed user-saved profile
  ) {

  var cols = [];
  for (var i = 0; i < 6; i++) {
    if (colMode <= 0) {
      cols[i] = color(bulbsCurrentHSB[i]);
    } else if (colMode > 0) {
      var prof	= savedFavList.searchNodeAt(colMode);
      cols		= prof.data;
    }
  }
  noStroke();
  switch(nz) {    

  case 3:
    {
      //var ptsX = []; //= new float[7];
      //var ptsY = []; //= new float[7];
      var tmx = dBias*(gx/dBias);
      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        var tmX = [gx, gx+dx, gx+dx, gx+dx-dx*0.333, gx-dx+dx*0.333, gx-dx, gx-dx];
        var tmY = [gy-dy, gy-dy, gy+dy, gy+dy, gy+dy, gy+dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {          
        var tmX = [gx+dx, gx+dx, gx-dx, gx-dx, gx-dx, gx-dx, gx+dx];
        var tmY = [gy, gy+dy, gy+dy, gy+dy-dy*0.333, gy-dy+dy*0.333, gy-dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        var tmX = [gx, gx-dx, gx-dx, gx-dx+dx*0.333, gx+dx-dx*0.333, gx+dx, gx+dx];
        var tmY = [gy+dy, gy+dy, gy-dy, gy-dy, gy-dy, gy-dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {          
        var tmX = [gx-dx, gx-dx, gx+dx, gx+dx, gx+dx, gx+dx, gx-dx];
        var tmY = [gy, gy-dy, gy-dy, gy-dy+dy*0.333, gy+dy-dy*0.333, gy+dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        var tmX = [gx+dx, gx+dx, gx-dx/4, gx-dx, gx-dx, gx-dx, 0];
        var tmY = [gy-dy, gy+dy, gy+dy, gy+dy, gy+dy/4, gy-dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        var tmX = [gx+dx, gx-dx, gx-dx, gx-dx, gx-dx/4, gx+dx, 0];
        var tmY = [gy+dy, gy+dy, gy-dy/4, gy-dy, gy-dy, gy-dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        var tmX = [gx-dx, gx-dx, gx+dx/4, gx+dx, gx+dx, gx+dx, 0];
        var tmY = [gy+dy, gy-dy, gy-dy, gy-dy, gy-dy/4, gy+dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        var tmX = [gx-dx, gx+dx, gx+dx, gx+dx, gx+dx/4, gx-dx, 0];
        var tmY = [gy-dy, gy-dy, gy+dy/4, gy+dy, gy+dy, gy+dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(cols[0]);
        quad(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(cols[1]);
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(cols[2]);
        quad(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5], ptsX[6], ptsY[6]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(cols[0]);
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(cols[1]);
        quad(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(cols[2]);
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      }
    }
    break;

  case 4: 
    {
      //var ptsX = new float[6];
      //var ptsY = new float[6];

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        var tmX = [gx, gx+dx, gx+dx, gx, gx-dx, gx-dx];
        var tmY = [gy-dy, gy-dy, gy+dy, gy+dy, gy+dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {          
        var tmX = [gx+dx, gx+dx, gx-dx, gx-dx, gx-dx, gx+dx];
        var tmY = [gy, gy+dy, gy+dy, gy, gy-dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        var tmX = [gx, gx-dx, gx-dx, gx, gx+dx, gx+dx];
        var tmY = [gy+dy, gy+dy, gy-dy, gy-dy, gy-dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {          
        var tmX = [gx-dx, gx-dx, gx+dx, gx+dx, gx+dx, gx-dx];
        var tmY = [gy, gy-dy, gy-dy, gy, gy+dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        var tmX = [gx+dx, gx+dx, gx, gx-dx, gx-dx, gx-dx];
        var tmY = [gy-dy, gy+dy, gy+dy, gy+dy, gy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        var tmX = [gx+dx, gx-dx, gx-dx, gx-dx, gx, gx+dx];
        var tmY = [gy+dy, gy+dy, gy, gy-dy, gy-dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        var tmX = [gx-dx, gx-dx, gx, gx+dx, gx+dx, gx+dx];
        var tmY = [gy+dy, gy-dy, gy-dy, gy-dy, gy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        var tmX = [gx-dx, gx+dx, gx+dx, gx+dx, gx, gx-dx];
        var tmY = [gy-dy, gy-dy, gy, gy+dy, gy+dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(cols[0]);
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(cols[1]);
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(cols[2]);
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(cols[3]);
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(cols[0]);
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(cols[1]);
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(cols[2]);
        triangle(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(cols[3]);
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
      }
    }
    break;

  case 5:
    {
      //var ptsX = new float[9];
      //var ptsY = new float[9];

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5) {
        //             0      1      2        3        4        5      6      7        8
        var tmX = [gx, gx+dx, gx+dx, gx+dx, gx+dx/2, gx-dx/2, gx-dx, gx-dx, gx-dx];
        var tmY = [gy-dy, gy-dy, gy+dy/2, gy+dy, gy+dy, gy+dy, gy+dy, gy+dy/2, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 67.5  && ao <= 112.5) {
        //             0      1      2        3        4        5      6      7        8
        var tmX = [gx+dx, gx+dx, gx-dx/2, gx-dx, gx-dx, gx-dx, gx-dx, gx-dx/2, gx+dx];
        var tmY = [gy, gy+dy, gy+dy, gy+dy, gy+dy/2, gy-dy/2, gy-dy, gy-dy, gy-dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 157.5 && ao <= 202.5) {
        //             0      1      2        3        4        5      6      7        8
        var tmX = [gx, gx-dx, gx-dx, gx-dx, gx-dx/2, gx+dx/2, gx+dx, gx+dx, gx+dx];
        var tmY = [gy+dy, gy+dy, gy-dy/2, gy-dy, gy-dy, gy-dy, gy-dy, gy-dy/2, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 247.5 && ao <= 292.5) {
        //             0      1      2        3        4        5      6      7        8
        var tmX = [gx-dx, gx-dx, gx+dx/2, gx+dx, gx+dx, gx+dx, gx+dx, gx+dx/2, gx-dx];
        var tmY = [gy, gy-dy, gy-dy, gy-dy, gy-dy/2, gy+dy/2, gy+dy, gy+dy, gy+dy];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5) {
        //             0      1      2        3        4      5        6        7       8
        var tmX = [gx+dx, gx+dx, gx+dx/3, gx-dx/2, gx-dx, gx-dx, gx-dx, gx-dx, 0];
        var tmY = [gy-dy, gy+dy, gy+dy, gy+dy, gy+dy, gy+dy/2, gy-dy/3, gy-dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 112.5 && ao <= 157.5) {
        //             0      1      2        3        4      5        6        7       8
        var tmX = [gx+dx, gx-dx, gx-dx, gx-dx, gx-dx, gx-dx/2, gx+dx/3, gx+dx, 0];
        var tmY = [gy+dy, gy+dy, gy+dy/3, gy-dy/2, gy-dy, gy-dy, gy-dy, gy-dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 202.5 && ao <= 247.5) {
        //             0      1      2        3        4      5        6        7       8
        var tmX = [gx-dx, gx-dx, gx-dx/3, gx+dx/2, gx+dx, gx+dx, gx+dx, gx+dx, 0];
        var tmY = [gy+dy, gy-dy, gy-dy, gy-dy, gy-dy, gy-dy/2, gy+dy/3, gy+dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }
      if (ao > 292.5 && ao <= 337.5) {
        //             0      1      2        3        4      5        6        7       8
        var tmX = [gx-dx, gx+dx, gx+dx, gx+dx, gx+dx, gx+dx/2, gx-dx/3, gx-dx, 0];
        var tmY = [gy-dy, gy-dy, gy-dy/3, gy+dy/2, gy+dy, gy+dy, gy+dy, gy+dy, 0];
        ptsX = tmX;
        ptsY = tmY;
      }

      // Hori/Vert Cases
      if (ao > 337.5 && ao <= 360.0 || ao >= 0 && ao < 22.5  || 
        ao > 67.5  && ao <= 112.5 ||
        ao > 157.5 && ao <= 202.5 ||
        ao > 247.5 && ao <= 292.5) {
        fill(cols[0]);
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(cols[1]);
        quad(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3], ptsX[4], ptsY[4]);
        fill(cols[2]);
        triangle(ptsX[0], ptsY[0], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
        fill(cols[3]);
        quad(ptsX[0], ptsY[0], ptsX[5], ptsY[5], ptsX[6], ptsY[6], ptsX[7], ptsY[7]);
        fill(cols[4]);
        triangle(ptsX[0], ptsY[0], ptsX[7], ptsY[7], ptsX[8], ptsY[8]);
      } else 
      // Diagonal Cases
      if (ao > 22.5 && ao <= 67.5  ||
        ao > 112.5 && ao <= 157.5 ||
        ao > 202.5 && ao <= 247.5 ||
        ao > 292.5 && ao <= 337.5) {
        fill(cols[0]);
        triangle(ptsX[0], ptsY[0], ptsX[1], ptsY[1], ptsX[2], ptsY[2]);
        fill(cols[1]);
        triangle(ptsX[0], ptsY[0], ptsX[2], ptsY[2], ptsX[3], ptsY[3]);
        fill(cols[2]);
        quad(ptsX[0], ptsY[0], ptsX[3], ptsY[3], ptsX[4], ptsY[4], ptsX[5], ptsY[5]);
        fill(cols[3]);
        triangle(ptsX[0], ptsY[0], ptsX[5], ptsY[5], ptsX[6], ptsY[6]);
        fill(cols[4]);
        triangle(ptsX[0], ptsY[0], ptsX[6], ptsY[6], ptsX[7], ptsY[7]);
      }
    }
    break;

  default:
    drawHomeCircle(gx, gy, dx, dy, nz, ao-90, colMode);
    break;
  }
}
