function touchStarted() {
  //necessarily empty
}

/*
function watchPoint(px, py, pr) {
  var wasMousePressed = false;
  if (1 >= pow((mouseX-px), 2) / pow(pr/2, 2) + pow((mouseY - py), 2) / pow(pr/2, 2)) 
  {
    if (touches.length >= 1 || isMousePressed) {
      wasMousePressed	= true;
    } else 
    if (touches.length == 0 && !isMousePressed) {
      wasMousePressed	= false;
    }
  }
  return wasMousePressed;
}
*/

class dropMenu {
  constructor(
	mx,								//float, menu x-coordinate
	my,								//float, menu y-coordinate
	ms,								//float, menu size scalar (0-1)
	angLimit,						//float, radians, icon rotation upon menu opening
	dWidth,							//int,	 width of display
	dHeight,						//int,	 height of display
	dBias,							//float, display bias, for dpi scaling relative coords
	frameLimit, 					//int,   duration, in frames, of animations
	numElements,					//uint,  number of elements to iterate over
	touchOpen,						//bool,  determines whether menu is touched to open
	dir,							//string, 'UP', 'DOWN', 'LEFT', 'RIGHT'
	) {
	this.mx				= mx;
	this.my				= my;
	this.ms				= ms;
	this.angLimit		= angLimit;
	this.width			= dWidth;
	this.height			= dHeight;
	this.dBias			= dBias;
	this.frameLimit		= frameLimit;
	this.numElements	= numElements;
	this.touchOpen		= touchOpen;
	this.dir			= dir;
	this.selection		= 1;
	this.mc				= 0;				//int,	animation cursor for menu open
	this.pc				= 0;				//int,	animation cursor for scroll effect
	this.scroll			= 0;				//int,	scroll direction (pd)
	this.menuOpen		= false;
	this.touchHold		= false;
	this.diff			= 1;

	this.het			= -this.dBias;
	this.tm04			= this.dBias*0.04;
	this.tm05			= this.dBias*0.05;
	this.tm06 			= this.dBias*0.06;
	this.tm08 			= this.dBias*0.08;
	this.tm10			= this.dBias*0.10;
	this.tm12			= this.dBias*0.12;
	this.tm13 			= this.dBias*0.125;
	this.tm15 			= this.dBias*0.15;
	this.tm20			= this.dBias*0.2;
	this.tm25			= this.dBias*0.25;
	this.tm30			= this.dBias*0.3;
	this.tm35			= this.dBias*0.35;
	this.tm40			= this.dBias*0.4;
  }

  isOpen()	 { return this.menuOpen		}
  getSel()	 { return this.selection	}
  getSelX()	 { return 0					}
  getSelY()	 { return 0					}
  getSelS()	 { return 1					}
  getX()	 { return this.mx			}
  getY()	 { return this.my			}
  setX(xPos) { this.mx = xPos			}
  setY(yPos) { this.my = yPos			}

  //Detect User Input Helper
  watchPoint(px, py, pr) {
	if (touches.length == 0 && !isMousePressed)
	  this.interaction = false;
	if (touches.length >= 1 || isMousePressed)
	  this.interaction = true;

	var wasMousePressed = false;
	if (1 >= pow((mouseX-px), 2) / pow(pr/2, 2) + pow((mouseY - py), 2) / pow(pr/2, 2)) 
	{
	  if (touches.length >= 1 || isMousePressed) {
		wasMousePressed	= true;
	  } else 
	  if (touches.length == 0 && !isMousePressed) {
		wasMousePressed	= false;
	  }
	}
	//console.log(wasMousePressed);
	return wasMousePressed;
  }

  //Animation Curve Helper
  animCurve(c) {
	return -2.25*pow(c/(this.frameLimit*1.5), 2) + 1;
  }

  //Animation Curve Helper
  animCurveBounce(c) {
	if (c == this.frameLimit)
	  return 0;
	else
	  return -3*pow((c-(this.frameLimit*0.14167))/(this.frameLimit*1.5), 2)+1.02675926;
  }

  selActive(selection){
	var acif	= this.animCurve(this.frameLimit-this.mc);
	var tm35	= this.tm35;
	var het 	= this.het;
	var oy;
	var ox;
	switch(this.numElements) {
	case 1:
	  het *= 0.333;
	  break;

	case 2:
	  het *= 0.667;
	  break;

	default:
	  //oy = het;
	}

	if (this.dir == 'UP')	{ oy = het;   ox = 0;	}
	if (this.dir == 'DOWN')	{ oy = -het;  ox = 0; 	}
	if (this.dir == 'LEFT') { oy = 0;	  ox = het; }
	if (this.dir == 'RIGHT'){ oy = 0; 	  ox = -het;}

	switch(this.numElements) {
	  //Apply Selection if clicked
	  case 1:
		if (this.mc == this.frameLimit &&
		  this.watchPoint(this.mx+(ox*acif), this.my+(oy*acif), tm35)) {
		  return true;
		} else { return false; }
	  break;

	  case 2:
		if (this.mc == this.frameLimit) {
		  if (selection == 1 &&
		  this.watchPoint(this.mx+(ox*acif), this.my+(oy*acif), tm35))
			return true;
		  else if (selection == 2 &&
		  this.watchPoint(this.mx+(ox*0.5*acif), this.my+(oy*0.5*acif), tm35))
			return true;
		  else
			return false;
		}
/*
		if (this.mc == this.frameLimit &&
		  this.watchPoint(this.mx+(ox*acif), this.my+(oy*acif), tm35)) {
		  return true;
		} else //{ return false; }

		if (this.mc == this.frameLimit &&
		  this.watchPoint(this.mx+(ox*0.5*acif), this.my+(oy*0.5*acif), tm35)) {
		  return true;
		} else { return false; }
*/
	  break;
	
	  default:
		if (this.mc == this.frameLimit &&
		  this.pc == this.frameLimit &&
		  this.selection == selection &&
		  this.watchPoint(this.mx+(ox*0.67*acif), this.my+(oy*0.67*acif), tm35)) {
		  return true;
		} else { return false; }
	}
  }

  drawMenu() {
	var acibf	= this.animCurveBounce(this.frameLimit-this.mc);
	var acif	= this.animCurve	  (this.frameLimit-this.mc);
	var acibp	= this.animCurveBounce(this.frameLimit-this.pc);
	var acip	= this.animCurve	  (this.frameLimit-this.pc);
	var het 	= this.het;
	var tm04	= this.tm04;
	var tm05	= this.tm05;
	var tm06 	= this.tm06;
	var tm08 	= this.tm08;
	var tm10	= this.tm10;
	var tm12	= this.tm12;
	var tm13 	= this.tm13;
	var tm15 	= this.tm15;
	var tm20	= this.tm20;
	var tm25	= this.tm25;
	var tm30	= this.tm30;
	var tm35	= this.tm35;
	var tm40	= this.tm40;

/*
	var het 	= -this.dBias;
	var tm04	= this.dBias*0.04;
	var tm05	= this.dBias*0.05;
	var tm06 	= this.dBias*0.06;
	var tm08 	= this.dBias*0.08;
	var tm10	= this.dBias*0.10;
	var tm12	= this.dBias*0.12;
	var tm13 	= this.dBias*0.125;
	var tm15 	= this.dBias*0.15;
	var tm20	= this.dBias*0.2;
	var tm25	= this.dBias*0.25;
	var tm30	= this.dBias*0.3;
	var tm35	= this.dBias*0.35;
	var tm40	= this.dBias*0.4;
*/
	var oy;
	var ox;
	strokeWeight(1);
	stroke(96);
	fill(240);
	noStroke();
	ellipse(this.mx, this.my, tm40, tm40);  

	switch(this.numElements) {
	case 1:
	  het *= 0.333;
	  break;

	case 2:
	  het *= 0.667;
	  break;

	default:
	}

	if (this.dir == 'UP')	{ oy = het;   ox = 0;	}
	if (this.dir == 'DOWN')	{ oy = -het;  ox = 0; 	}
	if (this.dir == 'LEFT') { oy = 0;	  ox = het; }
	if (this.dir == 'RIGHT'){ oy = 0; 	  ox = -het;}

	ellipse(this.mx+ox*acif, this.my+oy*acif, tm40);
	if (this.dir == 'UP'	|| this.dir == 'DOWN')
	  rect(this.mx-tm20, this.my, tm40, oy*acif);
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	  rect(this.mx, this.my-tm20, ox*acif, tm40);
	stroke(96);
	noStroke();

	//Draw Heart Iconography
	var xrs 	= cos(PI-acibf*this.angLimit);
	var yrs 	= sin(PI-acibf*this.angLimit);

	fill(96);
	ellipse ( 
	  this.mx-tm06*xrs, 
	  this.my+tm06*yrs+tm08*acibf, 
	  tm15);

	ellipse ( 
	  this.mx+tm06*xrs, 
	  this.my-tm06*yrs+tm08*acibf, 
	  tm15);

	triangle (
	  this.mx-tm13*xrs-tm04*yrs, 
	  this.my+tm13*yrs+tm04*xrs-tm08*xrs+tm08*acibf, 
	  this.mx+tm13*xrs-tm04*yrs, 
	  this.my-tm13*yrs+tm04*xrs-tm08*xrs+tm08*acibf, 
	  this.mx-tm15*yrs, 
	  this.my-tm13*xrs+tm04*xrs-tm08*xrs+tm08*acibf);

	fill(96+(255-96)*this.animCurve(this.mc), (this.mc >= floor(this.frameLimit*0.25) ? 255 : 0));

	// Watch Button for input
	if (this.watchPoint(this.mx, this.my, tm40, tm40) && 
		!this.touchHold && 
		this.touchOpen) {
	  if (!this.menuOpen) {
		this.menuOpen = true;
	  } else {
		this.menuOpen = false;
	  }
	  this.touchHold = true;
	} else if (!this.interaction) {
	  this.touchHold = false;
	}

/*~~~~~~~~~~~~~~~~~~~~~~~~~~~Draw Menu for 1 Element~~~~~~~~~~~~~~~~~~~~~~~~~~~*/
	if (this.numElements == 1) {
	  ellipse(this.mx+ox*acif, this.my+(oy*acif), tm30);
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));

	  // Close Favorites if open and somewhere else on screen is touched
	  if (this.interaction &&
		this.menuOpen &&
		this.mc == this.frameLimit &&
		!this.touchHold &&
		!this.watchPoint(this.mx, this.my+(oy*acif), tm35)
      ) {
		this.menuOpen = false;
	  }
	}

  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 2 profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  else if (this.numElements == 2) {
    ellipse(this.mx+ox*acif, this.my+(oy*acif), tm30);
    ellipse(this.mx+ox*0.5*acif, this.my+(oy*0.5*acif), tm30);
	if (this.dir == 'UP'	|| this.dir == 'DOWN')
	  rect (this.mx-tm15, this.my+(oy*0.50*acif), tm30, (oy*0.50*acif));
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	  rect (this.mx+ox*0.5*acif, this.my-tm15, (ox*0.50*acif), tm30);
	
    fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));

    // Close Favorites if open and somewhere else on screen is touched
    if (this.interaction &&
      this.menuOpen &&
      this.mc == this.frameLimit &&
      !this.touchHold &&
      !this.watchPoint(this.mx, this.my+(oy*acif), tm35) &&
      !this.watchPoint(this.mx, this.my+(oy*0.5*acif), tm35)
      ) {
      this.menuOpen = false;
    }
  } else  

  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 3+ profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  if (this.numElements >=3) {
    var utri	= 1;
    var mtri	= 0.7;
    var ltri	= 0.4;
    ellipse(this.mx+(ox*acif), this.my+(oy*acif), tm30);
    ellipse(this.mx+(ox*0.4*acif), this.my+(oy*0.4*acif), tm30);

	// UP - DOWN
	if (this.dir == 'UP' || this.dir == 'DOWN')
	{
	  rect (this.mx-tm15, 
		this.my+(oy*acif), 
		dBias*.3, 
		-(oy*0.6*acif));
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));
	  ellipse(this.mx-tm04, this.my+(oy*mtri*acif), tm20);
	}

	// LEFT - RIGHT
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	{
	  rect (this.mx+(ox*acif), 
		this.my-tm15, 
		-(ox*0.6*acif),
		dBias*.3);
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));
	  ellipse(this.mx+(ox*mtri*acif), this.my-tm04, tm20);
	}

    if (this.menuOpen && this.pc == this.frameLimit) {
	  fill(0, 64, 255);
	  ellipse(
		this.mx+(ox*utri*acif),
		this.my+(oy*utri*acif),
		tm20*0.67);
	  fill(85*0, 64, 255);
	  ellipse(
		this.mx+(ox*mtri*acif),
		this.my+(oy*mtri*acif),
		tm20);
	  fill(170*0, 64, 255);
	  ellipse(
		this.mx+(ox*ltri*acif),
		this.my+(oy*ltri*acif),
		tm20*0.67);

    } else if (this.menuOpen && this.pc != this.frameLimit && this.scroll == -1) {
      //Scoll Up
	  fill(0, 64, 255);
	  ellipse(
		this.mx+(ox*utri*acif),
		this.my+(oy*utri*acif),
		tm20*0.67*(1-acip));

	  ellipse(
		this.mx+(ox*utri*acif)-ox*0.31*(1-acibp),
		this.my+(oy*utri*acif)-oy*0.31*(1-acibp),
		tm20*(0.67+0.33*(1-acibp)));
	  fill(85*0, 64, 255);
	  ellipse(
		this.mx+(ox*mtri*acif)-ox*0.31*(1-acibp),
		this.my+(oy*mtri*acif)-oy*0.31*(1-acibp),
		tm20*(0.67+0.33*(acibp)));
	  fill(170*0, 64, 255);
	  ellipse(
		this.mx+(ox*ltri*acif),
		this.my+(oy*ltri*acif),
		tm20*0.67*pow(acibp, 6));

    } else if (this.menuOpen && this.pc != this.frameLimit && this.scroll == 1) {
      //Scroll down
	  fill(0, 64, 255);
	  ellipse(
		this.mx+(ox*ltri*acif),
		this.my+(oy*ltri*acif),
		tm20*0.67*(1.0-acip));

	  ellipse(
		this.mx+(ox*ltri*acif)+ox*0.31*(1-acibp),
		this.my+(oy*ltri*acif)+oy*0.31*(1-acibp),
		tm20*(0.67+0.33*(1-acibp)));
	  fill(85*0, 64, 255);
	  ellipse(
		this.mx+(ox*mtri*acif)+ox*0.31*(1-acibp),
		this.my+(oy*mtri*acif)+oy*0.31*(1-acibp),
		tm20*(0.67+0.33*(acibp)));
	  fill(170*0, 64, 255);
	  ellipse(
		this.mx+(ox*utri*acif),
		this.my+(oy*utri*acif),
		tm20*0.67*pow(acibp, 6));
    }

    // Close Favorites if open and somewhere else on screen is touched
    if (this.interaction &&
      this.menuOpen &&
      this.mc == this.frameLimit &&
      !this.touchHold &&
      !this.watchPoint(this.mx+(ox*acif), this.my+(oy*acif), tm35) &&
      !this.watchPoint(this.mx+(ox*0.4*acif), this.my+(oy*0.4*acif), tm35) &&
      !this.watchPoint(this.mx+(ox*0.7*acif), this.my+(oy*0.7*acif), tm35)
      ) {
      this.menuOpen = false;
    }

    if (this.watchPoint(this.mx+(ox*acif), this.my+(oy*acif), tm35) &&
      this.mc == this.frameLimit &&
      !this.touchHold) {
      this.menuOpen = true;
      //pt >= savedFavList.listSize() ? this.pt = 1 : this.pt += 1;
      this.selection >= this.numElements ? this.selection = 1 : this.selection += 1;
      this.touchHold = true;
      this.pc = 0;
      this.scroll = 1;
    }

    if (this.watchPoint(this.mx+(ox*0.4*acif), this.my+(oy*0.4*acif), tm35) &&
      this.mc == this.frameLimit &&
      !this.touchHold) {
      this.menuOpen = true;
      //pt <= 1 ? pt = savedFavList.listSize() : pt -= 1;
      this.selection <= 1 ? this.selection = this.numElements : this.selection -= 1;
      this.touchHold = true;
      this.pc = 0;
      this.scroll = -1;
    }

  }
	if (this.pc != this.frameLimit) {
	  this.pc = constrain(this.pc + 6*this.diff, 0, this.frameLimit);
	}

	if (this.menuOpen)
	  this.mc = constrain(this.mc + 6*this.diff, 0, this.frameLimit);
	else if (!this.menuOpen) 
	  this.mc = constrain(this.mc - 6*this.diff, 0, this.frameLimit);

  }
}
