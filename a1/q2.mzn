int: num_locations = 7;
int: num_neighbors = 15;

% The locations and their covered neighbors 
% A 2D Array
% locs[1][1] == 0 indicates that location 1 does not cover neighbor 1
% locs[1][1] == 1 indicates that location 1 covers neighbor 1
array[1..num_locations, 1..num_neighbors] of 0..1: ln = array2d(1..num_locations, 1..num_neighbors,
    [
      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
      1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0,
      0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0,
      0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0,
      0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0,
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1,
    ]);

% Which alternatives to choose (decision variable)
array[1..num_locations] of var 0..1: x;

% The objective to minimize
var int: obj = sum(i in 1..num_locations)(x[i]);

constraint forall(j in 1..num_neighbors)(sum(i in 1..num_locations)(x[i]*ln[i, j]) >= 1);

solve minimize obj;

output [
  "x:" ++ show(x) ++ "\n"
];