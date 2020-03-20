%Raeanne Ramharrack  816008940
%Tevin Achong        816000026

var int: C1W1;
var int: C1W2;
var int: C1W3;
var int: C1W4;
var int: C2W1;
var int: C2W2;
var int: C2W3;
var int: C2W4;
var int: C3W1;
var int: C3W2;
var int: C3W3;
var int: C3W4;

constraint C1W1 + C1W2 + C1W3 + C1W4 <= 75;    %Canary1 can supply at most 75
constraint C2W1 + C2W2 + C2W3 + C2W4 <= 125;   %Canary2 can supply at most 125
constraint C3W1 + C3W2 + C3W3 + C3W4 <= 100;   %Canary3 can supply at most 100

constraint C1W1 + C2W1 + C3W1 = 80;   %Warehouse1's demand must be met
constraint C1W2 + C2W2 + C3W2 = 65;   %Warehouse2's demand must be met
constraint C1W3 + C2W3 + C3W3 = 70;   %Warehouse3's demand must be met
constraint C1W4 + C2W4 + C3W4 = 85;   %Warehouse4's demand must be met

constraint C1W1 >= 0;
constraint C1W2 >= 0;
constraint C1W3 >= 0;
constraint C1W4 >= 0;
constraint C2W1 >= 0;
constraint C2W2 >= 0;
constraint C2W3 >= 0;
constraint C2W4 >= 0;
constraint C3W1 >= 0;
constraint C3W2 >= 0;
constraint C3W3 >= 0;
constraint C3W4 >= 0;

solve minimize C1W1 * 464 + C1W2 * 513 + C1W3 * 654 + C1W4 * 867 + C2W1 * 352 + C2W2 * 416 + C2W3 * 690 + C2W4 * 791 + C3W1 * 995 + C3W2 * 682 + C3W3 * 388 + C3W4 * 685;


