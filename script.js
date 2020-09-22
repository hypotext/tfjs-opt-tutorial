const RANGE = 10.0;
const NUM_STEPS = 1000;

// Use: open index.html in browser; check console

console.log("Computing grad of a 1D function");

// Define the function
// f(x) = x ^ 2
const f = x => x.square();

// Take its gradient
// f'(x) = 2x
const g = tf.grad(f);

// Evaluate the gradient at a specific point (one tensor)
const x = tf.tensor1d([2, 3]); // Note automatic size casting
console.log("f'(x) = ");
g(x).print();

// Make a variable to optimize the function on, and initialize it randomly in [0,10]
const a = tf.scalar(RANGE * Math.random()).variable();
console.log(`a (before opt): ${a.dataSync()}`);

// Make an optimizer with a specific descent algorithm and parameter
const learningRate = 0.1; // Note the learning rate really influences the rate of convergence!
const optimizer = tf.train.adam(learningRate);
// const optimizer = tf.train.sgd(learningRate);

// Optimize the function WRT the variable for NUM_STEPS iterations
let res;
for (let i = 0; i < NUM_STEPS; i++) {
  // `minimize` takes a closure of the function applied to the variable(s) to be optimized
  // It will mutate the variables to set its values
  // Note `minimize` also has params returnCost and varList (for specifying variables)
  res = optimizer.minimize(() => f(a), returnCost = true);
}

// Print the results
console.log(`a (after opt): ${a.dataSync()}`);
console.log(`f(a): ${res}`);
console.log(`----------------`);

// ------------------------------

console.log("Computing grad of a multivariate function");

// Define the function
// Note this is (x,y) not ([x,y])
let centerFn = (x, y) => x.square().add(y.square());

// Compute gradient (separately from optimization)
// Use `grads`, not `grad`, for multivariate function
const gradFn = tf.grads(centerFn);

// Define an example input
let num = x => tf.scalar(x, 'int32');
let varying_vals = [9.0, 25.0].map(num);

// Evaluate the gradient at the input
let [dx, dy] = gradFn(varying_vals);
// Note the x passed in grad(f)(x) must be a list of tensors
console.log("centerFn(varying_vals) = ");
dx.print();
dy.print();

// Optimize input params (c,d) to minimize the function
// Using the same optimizer from above
const c = tf.scalar(RANGE * Math.random()).variable();
const d = tf.scalar(RANGE * Math.random()).variable();
console.log(`before opt | c: ${c.dataSync()}, d: ${d.dataSync()}`);

let res2;
for (let i = 0; i < NUM_STEPS; i++) {
  res2 = optimizer.minimize(() => centerFn(c, d), returnCost = true);
}

// Print results
console.log(`after opt | c: ${c.dataSync()}, d: ${d.dataSync()}`);
console.log(`f(a): ${res2}`);

// tfjs also works for more complex functions, like function composition, etc. but doesn't deal with constraints
