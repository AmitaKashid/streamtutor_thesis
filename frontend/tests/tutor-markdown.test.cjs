const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');
const React = require('react');
const { renderToStaticMarkup } = require('react-dom/server');

const filename = path.resolve(__dirname, '../app/tutor-markdown.tsx');
const source = fs.readFileSync(filename, 'utf8');
const output = ts.transpileModule(source, { compilerOptions: {
  jsx: ts.JsxEmit.ReactJSX, module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020,
}}).outputText;
const compiled = new Module(filename, module);
compiled.filename = filename;
compiled.paths = Module._nodeModulePaths(path.dirname(filename));
compiled._compile(output, filename);
const { TutorMarkdown } = compiled.exports;

test('renders headings, emphasis and lists while escaping HTML', () => {
  const html = renderToStaticMarkup(React.createElement(TutorMarkdown, {
    text: '### Overview\nA **decision tree** predicts a class.\n- First step\n<script>alert(1)</script>',
  }));
  assert.match(html, /role="heading"[^>]*>Overview/);
  assert.match(html, /<strong>decision tree<\/strong>/);
  assert.match(html, /First step/);
  assert.ok(!html.includes('###'));
  assert.ok(!html.includes('**'));
  assert.ok(html.includes('&lt;script&gt;'));
  assert.ok(!html.includes('<script>'));
});
