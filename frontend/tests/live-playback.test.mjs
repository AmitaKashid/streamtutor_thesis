import test from 'node:test';
import assert from 'node:assert/strict';
import { createSynchronizedPlayback } from '../app/synchronized-playback.mjs';

test('streams PCM immediately, accepts late words after generation, and drains on output clock', async (t) => {
  const text = [], nodes = [], done = [];
  let output = 0;
  const ctx = {
    state:'running', currentTime:0, destination:{},
    getOutputTimestamp: () => ({contextTime:output}),
    createBufferSource() {
      const node = {connect(){}, start(at){this.at=at;}, stop(){}, onended:null};
      nodes.push(node); return node;
    },
  };
  const player = createSynchronizedPlayback({getContext:()=>ctx, onText:(_id,s)=>text.push(s),
    onDrained:(id)=>done.push(id), onError:(_id,e)=>assert.fail(e)});
  t.after(()=>player.clear());
  player.start('t');
  player.startSegment('t','0','Hello world. ','acoustic');
  player.enqueueAudio('t',{duration:.1},'',[],{segmentId:'0',offset:0,frames:100,rate:1000});
  assert.equal(nodes.length,1); // neither final PCM nor word timings exist yet
  player.enqueueAudio('t',{duration:.1},'',[],{segmentId:'0',offset:100,frames:100,rate:1000});
  assert.equal(nodes[1].at,.1);
  player.endSegment('t','0',200,1000);
  player.finish('t');
  player.updateWords('t','0',[{text:'Hello ',start:0,end:.07},{text:'world. ',start:.1,end:.18}]);
  output=.05;
  await new Promise(r=>setTimeout(r,30));
  assert.deepEqual(text,['Hello ']);
  output=.15;
  await new Promise(r=>setTimeout(r,30));
  assert.deepEqual(text,['Hello ','world. ']);
  ctx.currentTime=.2; nodes[0].onended(); nodes[1].onended();
  assert.deepEqual(done,[]);
  output=.2;
  await new Promise(r=>setTimeout(r,30));
  assert.deepEqual(done,['t']);
  player.updateWords('t','0',[{text:'late',start:0,end:0}]);
  assert.equal(text.join(''),'Hello world. ');
});
