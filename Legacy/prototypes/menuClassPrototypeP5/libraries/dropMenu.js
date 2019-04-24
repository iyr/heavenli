function touchStarted() {
  //necessarily empty
}

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
	this.frameLimit		= frameLimit < 4 ? 4 : frameLimit; //render bug occurs if <4
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

	switch(this.numElements) {
	case 1:
	  this.het *= 0.333;
	  break;

	case 2:
	  this.het *= 0.667;
	  break;

	default:
	  break;
	}
	
	this.ox;
	this.oy;

	if (this.dir == 'UP')	{ this.oy = this.het;   this.ox = 0;	}
	if (this.dir == 'DOWN')	{ this.oy = -this.het;  this.ox = 0; 	}
	if (this.dir == 'LEFT') { this.oy = 0;	  this.ox = this.het; }
	if (this.dir == 'RIGHT'){ this.oy = 0; 	  this.ox = -this.het;}

	/* ~ ~ ~ ~ variables for menu elements ~ ~ ~ ~ */
    this.utri			= 1;
    this.mtri			= 0.7;
    this.ltri			= 0.4;
	this.utrix			= this.mx+(this.ox*this.utri);
	this.utriy			= this.my+(this.oy*this.utri);
	this.mtrix			= this.mx+(this.ox*this.mtri);
	this.mtriy			= this.my+(this.oy*this.mtri);
	this.ltrix			= this.mx+(this.ox*this.ltri);
	this.ltriy			= this.my+(this.oy*this.ltri);
	this.utris			= this.tm20*0.67;
	this.mtris			= this.tm20;
	this.ltris			= this.tm20*0.67;

	/* ~ ~ ~ ~ variables for scroll animation ~ ~ ~ ~ */
	this.rtris			= 0;
	this.rtrix			= this.mx+this.ox;
	this.rtriy			= this.my+this.oy;
	this.rEle			= 1;

	/* = = = = END OF CONSTRUCTOR = = = = */
  }

  isOpen()	 { return this.menuOpen;	}
  getSel()	 { return this.selection;	}
	/*
  getSel(n)	 { 
	if (n == 0)
	  return //FILL ME IN
	else
	  return this.selection;	
	}
*/
  getRX()	 { return this.rtrix;		}
  getRY()	 { return this.rtriy;		}
  getRS()	 { return this.rtris;		}
  getX()	 { return this.mx;			}
  getY()	 { return this.my;			}
  setX(xPos) { this.mx = xPos;			}
  setY(yPos) { this.my = yPos;			}

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

  //Apply Selection if clicked helper
  selActive(selection){
	var acif	= this.animCurve(this.frameLimit-this.mc);

	switch(this.numElements) {
	  case 1:
		if (this.mc == this.frameLimit &&
		  this.watchPoint(
			this.mx+(this.ox*acif), 
			this.my+(this.oy*acif), 
			this.tm35)) {
		  return true;
		} else { return false; }
	  break;

	  case 2:
		if (this.mc == this.frameLimit) {
		  if (selection == 1 &&
		  this.watchPoint(
			this.mx+(this.ox*acif), 
			this.my+(this.oy*acif), 
			this.tm35))
			return true;
		  else if (selection == 2 &&
		  this.watchPoint(
			this.mx+(this.ox*0.5*acif), 
			this.my+(this.oy*0.5*acif), 
			this.tm35))
			return true;
		  else
			return false;
		}
	  break;
	
	  default:
		if (this.mc == this.frameLimit &&
		  this.pc == this.frameLimit &&
		  this.selection == selection &&
		  this.watchPoint(
			this.mx+(this.ox*0.67*acif), 
			this.my+(this.oy*0.67*acif), 
			this.tm35)) 
		{
		  return true;
		} else { return false; }
	}
  }

  getR() {
	if (this.scroll < 0) // Scroll up
	  return this.selection + 2 >= this.numElements ? 
		this.selection + 2 - this.numElements : 
		this.selection + 2;
	else
	if (this.scroll > 0) // Scroll down
	  return this.selection - 2 <= 1 ? 
		this.numElements - this.selection - 2 : 
		this.selection - 2;
  }

  getEX(selection) {
	if (this.numElements >= 3)
	switch(selection)
	  {
		case this.selection:
		  return this.mtrix;
		  break;

		case (this.selection >= this.numElements ? 1 : this.selection + 1):
		  return this.utrix;
		  break;

		case (this.selection <= 1 ? this.numElements : this.selection - 1):
		  return this.ltrix;
		  break;
		
		default:
		  if ((selection >= 1) && (selection < this.selection - 1))
			return this.ltrix;
		  else
		  if ((selection <= numElements) && (selection > this.selection + 1))
			return this.utrix;
		  break;
		}
	else if (this.numElements == 2)
	  switch(selection) 
		{
		  case 1:
			return this.utrix
			break;
		  case 2:
			return this.ltrix
			break;
		}
	else if (this.numElements == 1)
	  return this.utrix
  }

  getEY(selection){
	if (this.numElements >= 3)
	  switch(selection)
		{
		case this.selection:
		  return this.mtriy;
		  break;

		case (this.selection >= this.numElements ? 1 : this.selection + 1):
		  return this.utriy;
		  break;

		case (this.selection <= 1 ? this.numElements : this.selection - 1):
		  return this.ltriy;
		  break;
		
		default:
		  if ((selection >= 1) && (selection < this.selection - 1))
			return this.ltriy;
		  else
		  if ((selection <= numElements) && (selection > this.selection + 1))
			return this.utriy;
		  break;
		}
	else if (this.numElements == 2)
	  switch(selection) 
		{
		  case 1:
			return this.utriy
			break;
		  case 2:
			return this.ltriy
			break;
		}
	else if (this.numElements == 1)
	  return this.utriy
  }

  getES(selection){
	if (this.numElements >= 3)
	switch(selection)
		{
		case this.selection:
		  return this.mtris;
		  break;

		case (this.selection >= this.numElements ? 1 : this.selection + 1):
		  return this.utris;
		  break;

		case (this.selection <= 1 ? this.numElements : this.selection - 1):
		  return this.ltris;
		  break;
		
		default:
		  return 0;
		  break;
		}
	else if (this.numElements == 2)
	  switch(selection) 
		{
		  case 1:
			return this.utris
			break;
		  case 2:
			return this.ltris
			break;
		}
	else if (this.numElements == 1)
	  return this.utris;
  }

  drawMenu() 
  {
	var acibf	= this.animCurveBounce(this.frameLimit-this.mc);
	var acif	= this.animCurve	  (this.frameLimit-this.mc);
	var acibp	= this.animCurveBounce(this.frameLimit-this.pc);
	var acip	= this.animCurve	  (this.frameLimit-this.pc);

	strokeWeight(1);
	stroke(96);
	fill(240);
	noStroke();
	ellipse(this.mx, this.my, this.tm40, this.tm40);  
	ellipse(this.mx+this.ox*acif, this.my+this.oy*acif, this.tm40);
	if (this.dir == 'UP'	|| this.dir == 'DOWN')
	  rect(this.mx-this.tm20, this.my, this.tm40, this.oy*acif);
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	  rect(this.mx, this.my-this.tm20, this.ox*acif, this.tm40);
	stroke(96);
	noStroke();

	//Draw Heart Iconography
	var xrs 	= cos(PI-acibf*this.angLimit);
	var yrs 	= sin(PI-acibf*this.angLimit);

	fill(96);
	ellipse ( 
	  this.mx-this.tm06*xrs, 
	  this.my+this.tm06*yrs+this.tm08*acibf, 
	  this.tm15);

	ellipse ( 
	  this.mx+this.tm06*xrs, 
	  this.my-this.tm06*yrs+this.tm08*acibf, 
	  this.tm15);

	triangle (
	  this.mx-this.tm13*xrs-this.tm04*yrs, 
	  this.my+this.tm13*yrs+this.tm04*xrs-this.tm08*xrs+this.tm08*acibf, 
	  this.mx+this.tm13*xrs-this.tm04*yrs, 
	  this.my-this.tm13*yrs+this.tm04*xrs-this.tm08*xrs+this.tm08*acibf, 
	  this.mx-this.tm15*yrs, 
	  this.my-this.tm13*xrs+this.tm04*xrs-this.tm08*xrs+this.tm08*acibf);

	fill(96+(255-96)*this.animCurve(this.mc), (this.mc >= floor(this.frameLimit*0.25) ? 255 : 0));

	// Watch Button for input
	if (this.watchPoint(this.mx, this.my, this.tm40, this.tm40) && 
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
		this.utrix = this.mx+this.ox*acif;
		this.utriy = this.my+this.oy*acif;
		this.utris = this.tm30;
	  ellipse(
		this.utrix,
		this.utriy, 
		this.utris);
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));
	  

	  // Close Favorites if open and somewhere else on screen is touched
	  if (this.interaction &&
		this.menuOpen &&
		this.mc == this.frameLimit &&
		!this.touchHold &&
		!this.watchPoint(this.mx, this.my+(this.oy*acif), this.tm35)
      ) {
		this.menuOpen = false;
	  }
	}

  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 2 profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  else if (this.numElements == 2) {
	this.utrix = this.mx+this.ox*acif;
	this.utriy = this.my+this.oy*acif;
	this.utris = this.tm30;
	this.ltrix = this.mx+this.ox*0.5*acif;
	this.ltriy = this.my+this.oy*0.5*acif;
	this.ltris = this.tm30;
	ellipse(
	  this.utrix,
	  this.utriy,
	  this.utris);
	ellipse(
	  this.ltrix,
	  this.ltriy,
	  this.ltris);
/*
    ellipse(this.mx+ox*acif, this.my+(this.oy*acif), this.tm30);
    ellipse(this.mx+ox*0.5*acif, this.my+(this.oy*0.5*acif), this.tm30);
*/
	if (this.dir == 'UP'	|| this.dir == 'DOWN')
	  rect (this.mx-this.tm15, this.my+this.oy*0.5*acif, this.tm30, (this.oy*0.50*acif));
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	  rect (this.mx+ox*0.5*acif, this.my-this.tm15, (this.ox*0.50*acif), this.tm30);
	
    fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));

    // Close Favorites if open and somewhere else on screen is touched
    if (this.interaction &&
      this.menuOpen &&
      this.mc == this.frameLimit &&
      !this.touchHold &&
      !this.watchPoint(this.mx, this.my+(this.oy*acif), this.tm35) &&
      !this.watchPoint(this.mx, this.my+(this.oy*0.5*acif), this.tm35)
      ) {
      this.menuOpen = false;
    }
  } else  

  /*~~~~~~~~~~~~~~~~~~~~~~Draw Favorites for 3+ profiles~~~~~~~~~~~~~~~~~~~~~~~~~ */
  if (this.numElements >=3) {
    ellipse(this.mx+(this.ox*acif), this.my+(this.oy*acif), this.tm30);
    ellipse(this.mx+(this.ox*0.4*acif), this.my+(this.oy*0.4*acif), this.tm30);

	// UP - DOWN
	if (this.dir == 'UP' || this.dir == 'DOWN')
	{
	  rect (this.mx-this.tm15, 
		this.my+(this.oy*acif), 
		dBias*.3, 
		-(this.oy*0.6*acif));
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));
	  ellipse(this.mx-this.tm04, this.my+(this.oy*this.mtri*acif), this.tm20);
	}

	// LEFT - RIGHT
	if (this.dir == 'RIGHT' || this.dir == 'LEFT')
	{
	  rect (this.mx+(this.ox*acif), 
		this.my-this.tm15, 
		-(this.ox*0.6*acif),
		dBias*.3);
	  fill(255, (this.mc >= floor(this.frameLimit*0.5) ? 255 : 0));
	  ellipse(this.mx+(this.ox*this.mtri*acif), this.my-this.tm04, this.tm20);
	}

    if (this.menuOpen && this.pc == this.frameLimit) {
	  this.utrix = this.mx+(this.ox*this.utri*acif);
	  this.utriy = this.my+(this.oy*this.utri*acif);
	  this.mtrix = this.mx+(this.ox*this.mtri*acif);
	  this.mtriy = this.my+(this.oy*this.mtri*acif);
	  this.ltrix = this.mx+(this.ox*this.ltri*acif);
	  this.ltriy = this.my+(this.oy*this.ltri*acif);
	  this.utris = this.tm20*0.67;
	  this.mtris = this.tm20;
	  this.ltris = this.tm20*0.67;
	  this.rtris = 0;
	  this.rtrix = this.mx+this.ox;
	  this.rtriy = this.my+this.oy;

	  fill(0, 64, 255);
	  ellipse(
		this.utrix,
		this.utriy,
		this.utris);
	  ellipse(
		this.mtrix,
		this.mtriy,
		this.mtris);
	  ellipse(
		this.ltrix,
		this.ltriy,
		this.ltris);

    } else if (this.menuOpen && this.pc != this.frameLimit && this.scroll == -1) {
      //Scoll Up
	  this.utrix = this.mx+(this.ox*this.utri*acif)-this.ox*0.31*(1-acibp);
	  this.utriy = this.my+(this.oy*this.utri*acif)-this.oy*0.31*(1-acibp);
	  this.mtrix = this.mx+(this.ox*this.mtri*acif)-this.ox*0.31*(1-acibp);
	  this.mtriy = this.my+(this.oy*this.mtri*acif)-this.oy*0.31*(1-acibp);
	  this.ltrix = this.mx+(this.ox*this.ltri*acif);
	  this.ltriy = this.my+(this.oy*this.ltri*acif);
	  this.utris = this.tm20*(0.67+0.33*(1-acibp));
	  this.mtris = this.tm20*(0.67+0.33*(acibp));
	  this.ltris = this.tm20*0.67*pow(acibp, 6);
	  this.rtrix = this.mx+(this.ox*this.utri*acif);
	  this.rtriy = this.my+(this.oy*this.utri*acif);
	  this.rtris = this.tm20*0.67*(1-acip);

	  fill(0, 64, 255);
/*
	  ellipse(
		this.rtrix,
		this.rtriy,
		this.rtris);
*/
	  ellipse(
		this.utrix,
		this.utriy,
		this.utris);
	  ellipse(
		this.mtrix,
		this.mtriy,
		this.mtris);
	  ellipse(
		this.ltrix,
		this.ltriy,
		this.ltris);

    } else if (this.menuOpen && this.pc != this.frameLimit && this.scroll == 1) {
      //Scroll down
	  this.utrix = this.mx+(this.ox*this.utri*acif);
	  this.utriy = this.my+(this.oy*this.utri*acif);
	  this.mtrix = this.mx+(this.ox*this.mtri*acif)+this.ox*0.31*(1-acibp);
	  this.mtriy = this.my+(this.oy*this.mtri*acif)+this.oy*0.31*(1-acibp);
	  this.ltrix = this.mx+(this.ox*this.ltri*acif)+this.ox*0.31*(1-acibp);
	  this.ltriy = this.my+(this.oy*this.ltri*acif)+this.oy*0.31*(1-acibp);
	  this.utris = this.tm20*0.67*pow(acibp, 6);
	  this.mtris = this.tm20*(0.67+0.33*(acibp));
	  this.ltris = this.tm20*(0.67+0.33*(1-acibp));
	  this.rtrix = this.mx+(this.ox*this.ltri*acif);
	  this.rtriy = this.my+(this.oy*this.ltri*acif);
	  this.rtris = this.tm20*0.67*(1.0-acip);

	  fill(0, 64, 255);
/*
	  ellipse(
		this.rtrix,
		this.rtriy,
		this.rtris);
*/
	  ellipse(
		this.ltrix,  
		this.ltriy,
		this.ltris);
	  ellipse(
		this.mtrix,
		this.mtriy,
		this.mtris);
	  ellipse(
		this.utrix,
		this.utriy,
		this.utris);
    }

    // Close Favorites if open and somewhere else on screen is touched
    if (this.interaction &&
      this.menuOpen &&
      this.mc == this.frameLimit &&
      !this.touchHold &&
      !this.watchPoint(this.mx+(this.ox*acif), this.my+(this.oy*acif), this.tm35) &&
      !this.watchPoint(this.mx+(this.ox*0.4*acif), this.my+(this.oy*0.4*acif), this.tm35) &&
      !this.watchPoint(this.mx+(this.ox*0.7*acif), this.my+(this.oy*0.7*acif), this.tm35)
      ) {
      this.menuOpen = false;
    }

    if (this.watchPoint(this.mx+(this.ox*acif), this.my+(this.oy*acif), this.tm35) &&
      this.mc == this.frameLimit &&
      !this.touchHold) {
      this.menuOpen = true;
      //pt >= savedFavList.listSize() ? this.pt = 1 : this.pt += 1;
      this.selection >= this.numElements ? this.selection = 1 : this.selection += 1;
      this.touchHold = true;
      this.pc = 0;
      this.scroll = 1;
    }

    if (this.watchPoint(this.mx+(this.ox*0.4*acif), this.my+(this.oy*0.4*acif), this.tm35) &&
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
