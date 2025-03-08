"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        if (typeof b !== "function" && b !== null)
            throw new TypeError("Class extends value " + String(b) + " is not a constructor or null");
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
exports.MLP = exports.Layer = exports.Neuron = exports.Module = void 0;
var engine_1 = require("./engine");
var utils_1 = require("./utils");
// Base Module class
var Module = /** @class */ (function () {
    function Module() {
    }
    Module.prototype.parameters = function () {
        return [];
    };
    Module.prototype.call_value_ = function (x) {
        return [new engine_1.Value(0.0)];
    };
    Module.prototype.call_number_ = function (x) {
        var input = Array.from((0, utils_1.range)(0, x.length), function (i) {
            return (0, engine_1.ensureValue)(x[i]);
        });
        return this.call_value_(input);
    };
    Module.prototype.call = function (x) {
        if (typeof x[0] === 'number') {
            return this.call_number_(x);
        }
        else {
            return this.call_value_(x);
        }
    };
    Module.prototype.zero_grad = function () {
        this.parameters().forEach(function (v) {
            v.grad = 0;
        });
    };
    return Module;
}());
exports.Module = Module;
// Neuron Class
var Neuron = /** @class */ (function (_super) {
    __extends(Neuron, _super);
    function Neuron(nin, nonlin) {
        if (nonlin === void 0) { nonlin = true; }
        var _this = _super.call(this) || this;
        _this.w = Array.from((0, utils_1.range)(0, nin), function (x) { return new engine_1.Value(Math.random() * 2.0 - 1.0); });
        _this.b = new engine_1.Value(0.0);
        _this.nonlin = nonlin;
        return _this;
    }
    Neuron.prototype.call_value_ = function (x) {
        if (x.length != this.w.length) {
            throw new Error('Different sizes');
        }
        var act = this.w
            .map(function (e, i) {
            return e.mul(x[i]);
        })
            .reduce(function (sum, current) { return sum.add(current); }, new engine_1.Value(0.0))
            .add(this.b);
        return this.nonlin ? [act.relu()] : [act];
    };
    Neuron.prototype.parameters = function () {
        return this.w.concat([this.b]);
    };
    Neuron.prototype.toString = function () {
        return "".concat(this.nonlin ? 'ReLU' : 'Linear', " Neuron(").concat(this.w.length, ")");
    };
    return Neuron;
}(Module));
exports.Neuron = Neuron;
// Layer Class
var Layer = /** @class */ (function (_super) {
    __extends(Layer, _super);
    function Layer(nin, nout, nonlin) {
        if (nonlin === void 0) { nonlin = true; }
        var _this = _super.call(this) || this;
        _this.neurons = Array.from((0, utils_1.range)(0, nout), function (x) { return new Neuron(nin, nonlin); });
        return _this;
    }
    Layer.prototype.call_value_ = function (x) {
        var output = Array.from(this.neurons, function (n) { return n.call(x)[0]; });
        return output;
    };
    Layer.prototype.parameters = function () {
        var result = [];
        for (var _i = 0, _a = this.neurons; _i < _a.length; _i++) {
            var neuron = _a[_i];
            for (var _b = 0, _c = neuron.parameters(); _b < _c.length; _b++) {
                var param = _c[_b];
                result.push(param);
            }
        }
        return result;
    };
    Layer.prototype.toString = function () {
        return 'Layer';
    };
    return Layer;
}(Module));
exports.Layer = Layer;
// MLP Class
var MLP = /** @class */ (function (_super) {
    __extends(MLP, _super);
    function MLP(nin, nouts) {
        var _this = _super.call(this) || this;
        var sizes = [nin].concat(nouts);
        _this.layers = Array.from(nouts.keys(), function (i) { return new Layer(sizes[i], sizes[i + 1], i !== nouts.length - 1); });
        return _this;
    }
    MLP.prototype.call_value_ = function (x) {
        var result = this.layers[0].call(x);
        for (var i = 1; i < this.layers.length; i++) {
            result = this.layers[i].call(result);
        }
        return result;
    };
    MLP.prototype.parameters = function () {
        var result = [];
        for (var _i = 0, _a = this.layers; _i < _a.length; _i++) {
            var layer = _a[_i];
            for (var _b = 0, _c = layer.parameters(); _b < _c.length; _b++) {
                var param = _c[_b];
                result.push(param);
            }
        }
        return result;
    };
    MLP.prototype.toString = function () {
        return 'MLP';
    };
    return MLP;
}(Module));
exports.MLP = MLP;
