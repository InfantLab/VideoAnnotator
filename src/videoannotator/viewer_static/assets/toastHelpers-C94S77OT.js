import{j as i,aL as n}from"./index-Dx-ThYz7.js";function s(e){return i.jsx("button",{onClick:()=>{navigator.clipboard.writeText(e),n({title:"Copied!",description:"Error message copied to clipboard",duration:2e3})},className:"inline-flex h-8 shrink-0 items-center justify-center rounded-md border border-muted/40 bg-transparent px-3 text-sm font-medium hover:bg-destructive/10 focus:outline-none focus:ring-2 focus:ring-ring",children:"Copy"})}function c(e,t){const o=t.hint?`${t.message}

💡 Tip: ${t.hint}`:t.message,r=`Error

${o}`;e({title:"Error",description:o,variant:"destructive",duration:1e4,action:s(r)})}export{s as c,c as s};
