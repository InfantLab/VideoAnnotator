import{j as r,bi as n}from"./index-BLHipTmr.js";function s(e){return r.jsx("button",{onClick:()=>{navigator.clipboard.writeText(e),n({title:"Copied!",description:"Error message copied to clipboard",duration:2e3})},className:"inline-flex h-8 shrink-0 items-center justify-center rounded-md border border-muted/40 bg-transparent px-3 text-sm font-medium hover:bg-destructive/10 focus:outline-none focus:ring-2 focus:ring-ring",children:"Copy"})}function c(e,t){const i=t.hint?`${t.message}

💡 Tip: ${t.hint}`:t.message,o=`Error

${i}`;e({title:"Error",description:i,variant:"destructive",duration:1e4,action:s(o)})}export{s as c,c as s};
