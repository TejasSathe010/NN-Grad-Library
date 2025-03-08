export declare function ensureValue(x: number | Value): Value;
export declare class Value {
    data: number;
    children: Value[];
    op: string;
    _backward: () => void;
    grad: number;
    constructor(data: number, children?: Value[], op?: string);
    add(other_: Value | number): Value;
    mul(other_: Value | number): Value;
    pow(other_: number): Value;
    relu(): Value;
    backward(): void;
    toString(): string;
}
