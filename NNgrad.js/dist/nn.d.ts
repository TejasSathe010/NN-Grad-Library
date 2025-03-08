import { Value } from './engine';
export declare class Module {
    parameters(): Value[];
    call_value_(x: Value[]): Value[];
    call_number_(x: number[]): Value[];
    call(x: Value[] | number[]): Value[];
    zero_grad(): void;
}
export declare class Neuron extends Module {
    w: Value[];
    b: Value;
    nonlin: boolean;
    constructor(nin: number, nonlin?: boolean);
    call_value_(x: Value[]): Value[];
    parameters(): Value[];
    toString(): string;
}
export declare class Layer extends Module {
    neurons: Neuron[];
    constructor(nin: number, nout: number, nonlin?: boolean);
    call_value_(x: Value[]): Value[];
    parameters(): Value[];
    toString(): string;
}
export declare class MLP extends Module {
    layers: Layer[];
    constructor(nin: number, nouts: number[]);
    call_value_(x: Value[]): Value[];
    parameters(): Value[];
    toString(): string;
}
