import { Value, ensureValue } from './engine';
import { range } from './utils';

// Base Module class
export class Module {
  parameters(): Value[] {
    return [];
  }

  call_value_(x: Value[]): Value[] {
    return [new Value(0.0)];
  }

  call_number_(x: number[]): Value[] {
    const input: Value[] = Array.from(range(0, x.length), i =>
      ensureValue(x[i]),
    );
    return this.call_value_(input);
  }

  call(x: Value[] | number[]): Value[] {
    if (typeof x[0] === 'number') {
      return this.call_number_(<number[]>x);
    } else {
      return this.call_value_(<Value[]>x);
    }
  }

  zero_grad(): void {
    this.parameters().forEach(function (v: Value) {
      v.grad = 0;
    });
  }
}

// Neuron Class
export class Neuron extends Module {
  w: Value[];
  b: Value;
  nonlin: boolean;
  activation: string;

  constructor(nin: number, nonlin = true, activation = 'relu') {
    super();
    this.w = Array.from(
      range(0, nin),
      x => new Value(Math.random() * 2.0 - 1.0),
    );
    this.b = new Value(0.0);
    this.nonlin = nonlin;
    this.activation = activation;
  }

  call_value_(x: Value[]): Value[] {
    if (x.length != this.w.length) {
      throw new Error('Different sizes');
    }
    const act = this.w
      .map((e, i) => e.mul(x[i]))
      .reduce((sum, current) => sum.add(current), new Value(0.0))
      .add(this.b);
    if (this.activation === 'relu') {
      return [act.relu()];
    } else if (this.activation === 'sigmoid') {
      return [act.sigmoid()];
    } else if (this.activation === 'tanh') {
      return [act.tanh()];
    } else if (this.activation === 'linear') {
      return [act];
    } else {
      return [act]; // default to linear if unknown
    }
  }

  parameters(): Value[] {
    return this.w.concat([this.b]);
  }

  toString(): string {
    return `${this.nonlin ? 'ReLU' : 'Linear'} Neuron(${this.w.length})`;
  }
}

// Layer Class
export class Layer extends Module {
  neurons: Neuron[];

  constructor(nin: number, nout: number, nonlin = true, activation: string) {
    super();
    this.neurons = Array.from(range(0, nout), x => new Neuron(nin, nonlin, activation));
  }

  call_value_(x: Value[]): Value[] {
    const output = Array.from(this.neurons, n => n.call(x)[0]);
    return output;
  }

  parameters(): Value[] {
    const result: Value[] = [];
    for (const neuron of this.neurons) {
      for (const param of neuron.parameters()) {
        result.push(param);
      }
    }
    return result;
  }

  toString(): string {
    return 'Layer';
  }
}

// MLP Class
export class MLP extends Module {
  layers: Layer[];
  
  constructor(nin: number, nouts: number[], activations: string[]) {
    super();
    const sizes = [nin].concat(nouts);
    this.layers = [];
    for (let i = 0; i < nouts.length; i++) {
      let act = 'relu';
      if (activations && activations[i]) {
        act = activations[i];
      } else {
        act = i === nouts.length - 1 ? 'sigmoid' : 'relu';
      }
      this.layers.push(new Layer(sizes[i], sizes[i + 1], i !== nouts.length - 1, act));
    }
  }

  call_value_(x: Value[]): Value[] {
    let result: Value[] = this.layers[0].call(x);
    for (let i = 1; i < this.layers.length; i++) {
      result = this.layers[i].call(result);
    }
    return result;
  }

  parameters(): Value[] {
    const result: Value[] = [];
    for (const layer of this.layers) {
      for (const param of layer.parameters()) {
        result.push(param);
      }
    }
    return result;
  }

  toString() {
    return 'MLP';
  }
}
