"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.range = range;
function range(start, end) {
    return Array.from(Array(end - start).keys()).map(function (v) { return start + v; });
}
