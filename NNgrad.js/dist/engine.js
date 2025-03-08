"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Value = void 0;
exports.ensureValue = ensureValue;
function isNumber(x) {
    return typeof x === 'number';
}
function isString(x) {
    return typeof x === 'string';
}
function isValue(x) {
    return x instanceof Value;
}
// Ensure value function
function ensureValue(x) {
    if (isNumber(x)) {
        return new Value(x);
    }
    else {
        return x;
    }
}
// Value Class
var Value = /** @class */ (function () {
    function Value(data, children, op) {
        if (children === void 0) { children = []; }
        if (op === void 0) { op = ''; }
        this.data = data;
        this.children = children;
        this.op = op;
        this._backward = function () {
            return null;
        };
        this.grad = 0.0;
    }
    // Operations on Value class
    Value.prototype.add = function (other_) {
        var self = this;
        var other = ensureValue(other_);
        var out = new Value(this.data + other.data, [this, other], '+');
        out._backward = function () {
            self.grad += out.grad;
            other.grad += out.grad;
        };
        return out;
    };
    Value.prototype.mul = function (other_) {
        var other = ensureValue(other_);
        var self = this;
        var out = new Value(this.data * other.data, [this, other], '*');
        out._backward = function () {
            self.grad += other.data * out.grad;
            other.grad += self.data * out.grad;
        };
        return out;
    };
    Value.prototype.pow = function (other_) {
        var other = other_;
        var self = this;
        var out = new Value(Math.pow(this.data, other), [this], '**' + other.toString());
        out._backward = function () {
            self.grad += other * Math.pow(self.data, other - 1) * out.grad;
        };
        return out;
    };
    Value.prototype.relu = function () {
        var self = this;
        var out = new Value(this.data < 0 ? 0.0 : this.data, [this], 'ReLU');
        out._backward = function () {
            self.grad += (out.data > 0.0 ? 1.0 : 0.0) * out.grad;
        };
        return out;
    };
    // Backpropagation method
    Value.prototype.backward = function () {
        var topo = [];
        var visited = new Set();
        var build_topo = function (v) {
            if (!visited.has(v)) {
                visited.add(v);
                for (var _i = 0, _a = v.children; _i < _a.length; _i++) {
                    var child = _a[_i];
                    build_topo(child);
                }
                topo.push(v);
            }
        };
        build_topo(this);
        this.grad = 1;
        topo
            .slice()
            .reverse()
            .forEach(function (v) {
            v._backward();
        });
    };
    Value.prototype.toString = function () {
        return "Value(data=".concat(this.data, ", grad=").concat(this.grad, ", op=").concat(this.op, ")");
    };
    return Value;
}());
exports.Value = Value;
