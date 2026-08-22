; Structural discovery fixture: an unmarked kernel with the paged-access
; signature (KV gather whose address cone passes through a block-table
; LOAD, constant stride in the innermost loop). Discovery must report it
; as a candidate under NTA_DISCOVERY_NOTES=1 and must NOT authorize
; anything. The sibling kernel takes the index as an ARGUMENT (no loaded
; index): same loop shape, no paged note.
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @paged_gather(ptr %table, ptr %kv, ptr %out, i32 %n) {
entry:
  %row = load i32, ptr %table, align 4
  %row.ext = sext i32 %row to i64
  %base = getelementptr float, ptr %kv, i64 %row.ext
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %addr = getelementptr float, ptr %base, i64 %i
  %v = load float, ptr %addr, align 4
  %sum = fadd float %acc, %v
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, 128
  br i1 %cond, label %loop, label %exit

exit:
  store float %sum, ptr %out, align 4
  ret void
}

define ptx_kernel void @argument_gather(i64 %row, ptr %kv, ptr %out) {
entry:
  %base = getelementptr float, ptr %kv, i64 %row
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %i.next, %loop ]
  %acc = phi float [ 0.0, %entry ], [ %sum, %loop ]
  %addr = getelementptr float, ptr %base, i64 %i
  %v = load float, ptr %addr, align 4
  %sum = fadd float %acc, %v
  %i.next = add i64 %i, 1
  %cond = icmp slt i64 %i.next, 128
  br i1 %cond, label %loop, label %exit

exit:
  store float %sum, ptr %out, align 4
  ret void
}
