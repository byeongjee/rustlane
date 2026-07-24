
use proc_macro2::Span;
use quote::ToTokens;
use syn::parse::{Parse, ParseStream};
use syn::spanned::Spanned;
use syn::{
    parse_quote, parse_quote_spanned, BinOp, Block, Error, Expr, ExprBinary, ExprBlock,
    ExprForLoop, ExprIf, ExprLoop, ExprWhile, Ident, Lifetime, Macro, Pat, RangeLimits, Stmt,
    Token, Type,
};

use crate::kernel::{check_reserved, widen_path, widen_type};
use crate::scan::{scan_block, scan_expr, ExitScan};


pub enum RetMode {
    Unit,
    Value(Box<Type>),
}

pub fn rewrite_body(block: Block, ret_mode: RetMode) -> (Vec<Stmt>, Vec<Error>) {
    let has_ret = scan_block(&block).ret;
    let mut rw = Rewriter {
        errors: Vec::new(),
        loops: Vec::new(),
        barriers: Vec::new(),
        ret_mode,
        has_ret_machinery: has_ret,
    };

    let mut stmts = block.stmts;
    let tail: Option<Expr> = match stmts.last() {
        Some(Stmt::Expr(e, None)) if is_tail_value(e) => {
            let Some(Stmt::Expr(e, _)) = stmts.pop() else {
                unreachable!()
            };
            Some(e)
        }
        _ => None,
    };

    let mut out: Vec<Stmt> = Vec::new();
    if has_ret {
        if let RetMode::Value(ty) = &rw.ret_mode {
            out.push(parse_quote!(
                let mut __ret: #ty = ::core::default::Default::default();
            ));
        }
        out.push(parse_quote!(
            let mut __fn = EnterLoopN::<N>::enter_loop_n(__exec);
        ));
    }

    let n = stmts.len();
    let trailing_work = tail.is_some() || has_ret;
    for (k, s) in stmts.into_iter().enumerate() {
        let following = k + 1 < n || trailing_work;
        rw.rewrite_stmt(s, following, &mut out);
    }

    match (has_ret, &rw.ret_mode, tail) {
        (true, RetMode::Value(_), Some(t)) => {
            let t2 = rw.rw(t);
            out.push(parse_quote!(__ret.masked_assign(__exec, #t2);));
            out.push(Stmt::Expr(parse_quote!(__ret), None));
        }
        (true, RetMode::Value(_), None) => {
            out.push(Stmt::Expr(parse_quote!(__ret), None));
        }
        (_, _, Some(t)) => {
            let t2 = rw.rw(t);
            out.push(Stmt::Expr(t2, None));
        }
        _ => {}
    }

    (out, rw.errors)
}

fn is_tail_value(e: &Expr) -> bool {
    !matches!(
        e,
        Expr::If(_)
            | Expr::While(_)
            | Expr::ForLoop(_)
            | Expr::Loop(_)
            | Expr::Macro(_)
            | Expr::Return(_)
            | Expr::Break(_)
            | Expr::Continue(_)
    )
}


struct Frame {
    loop_id: Ident,
    iter_id: Option<Ident>,
    cont_label: Option<Lifetime>,
    brk_label: Option<Lifetime>,
}

#[derive(Clone, Copy, PartialEq)]
enum BarrierKind {
    Unmasked,
    Foreach,
}

struct Barrier {
    kind: BarrierKind,
    loops_len: usize,
}

struct Rewriter {
    errors: Vec<Error>,
    loops: Vec<Frame>,
    barriers: Vec<Barrier>,
    ret_mode: RetMode,
    has_ret_machinery: bool,
}

const ALLOWED_MACROS: &str =
    "allowed kernel macros: `foreach!`, `foreach_2d!`, `foreach_tiled!`, `unmasked!`, `cif!`, `cwhile!`";

impl Rewriter {
    fn err(&mut self, tokens: &dyn ToTokens, msg: &str) {
        self.errors.push(Error::new_spanned(tokens, msg));
    }

    fn check_ident(&mut self, id: &Ident) {
        if let Some(msg) = check_reserved(id) {
            self.errors.push(Error::new(id.span(), msg));
        }
    }

    fn check_place_root(&mut self, e: &Expr) {
        if let Some(id) = place_root_ident(e) {
            if let Some(msg) = check_reserved(id) {
                self.errors.push(Error::new(id.span(), msg));
            }
        }
    }

    fn check_tokens_reserved(&mut self, e: &dyn ToTokens) {
        fn walk(ts: proc_macro2::TokenStream, errors: &mut Vec<Error>) {
            for tt in ts {
                match tt {
                    proc_macro2::TokenTree::Ident(id)
                        if id.to_string().starts_with("__") =>
                    {
                        errors.push(Error::new(
                            id.span(),
                            "identifiers starting with `__` are reserved by #[kernel] machinery",
                        ));
                    }
                    proc_macro2::TokenTree::Group(g) => walk(g.stream(), errors),
                    _ => {}
                }
            }
        }
        walk(e.to_token_stream(), &mut self.errors);
    }

    fn accessible_frame(&self) -> Option<&Frame> {
        let floor = self.barriers.last().map(|b| b.loops_len).unwrap_or(0);
        if self.loops.len() > floor {
            self.loops.last()
        } else {
            None
        }
    }

    fn refresh_after(&mut self, s: ExitScan, sp: Span, out: &mut Vec<Stmt>) {
        if let Some(f) = self.loops.last() {
            if s.br || s.ret {
                let lid = &f.loop_id;
                out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&#lid);));
            }
            if s.co {
                if let Some(iid) = &f.iter_id {
                    out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&#iid);));
                }
            }
        } else if s.ret && self.has_ret_machinery && self.barriers.is_empty() {
            out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&__fn);));
        }
    }


    fn rewrite_block(&mut self, b: Block) -> Block {
        Block {
            brace_token: b.brace_token,
            stmts: self.rewrite_stmts(b.stmts),
        }
    }

    fn rewrite_stmts(&mut self, stmts: Vec<Stmt>) -> Vec<Stmt> {
        let n = stmts.len();
        let mut out = Vec::new();
        for (k, s) in stmts.into_iter().enumerate() {
            self.rewrite_stmt(s, k + 1 < n, &mut out);
        }
        out
    }

    fn rewrite_stmt(&mut self, s: Stmt, following: bool, out: &mut Vec<Stmt>) {
        match s {
            Stmt::Local(mut l) => {
                match &mut l.pat {
                    Pat::Ident(pi) => {
                        self.check_ident(&pi.ident);
                        if pi.subpat.is_some() {
                            self.err(&l.pat, "`ident @ pattern` bindings are not supported in #[kernel]");
                        }
                    }
                    Pat::Wild(_) => {}
                    Pat::Type(pt) => {
                        if let Pat::Ident(pi) = &*pt.pat {
                            self.check_ident(&pi.ident);
                        } else if !matches!(&*pt.pat, Pat::Wild(_)) {
                            self.err(
                                &pt.pat,
                                "destructuring `let` patterns are not supported in #[kernel]; \
                                 bind a single identifier",
                            );
                        }
                        widen_type(&mut pt.ty);
                    }
                    other => {
                        self.err(
                            other,
                            "destructuring `let` patterns are not supported in #[kernel]; \
                             bind a single identifier",
                        );
                    }
                }
                if let Some(init) = &mut l.init {
                    if let Some((else_tok, _)) = &init.diverge {
                        self.err(else_tok, "`let ... else` is not supported in #[kernel]");
                    }
                    let e = std::mem::replace(&mut *init.expr, Expr::PLACEHOLDER);
                    *init.expr = self.rw(e);
                }
                out.push(Stmt::Local(l));
            }
            Stmt::Item(i) => {
                self.err(&i, "item definitions inside #[kernel] bodies are not supported");
            }
            Stmt::Macro(m) => {
                if !m.attrs.is_empty() {
                    self.err(&m.attrs[0], "attributes on kernel statements are not supported");
                }
                self.emit_macro(m.mac, following, out);
            }
            Stmt::Expr(e, semi) => self.rewrite_expr_stmt(e, semi.is_some(), following, out),
        }
    }

    fn rewrite_expr_stmt(&mut self, e: Expr, semi: bool, following: bool, out: &mut Vec<Stmt>) {
        match e {
            Expr::If(i) => self.emit_if_stmt(i, out),
            Expr::While(w) => self.emit_while(w, out),
            Expr::ForLoop(f) => self.emit_for(f, out),
            Expr::Loop(l) => self.emit_loop(l, out),
            Expr::Assign(a) => self.emit_assign(a, out),
            Expr::Binary(b) if assign_binop(&b.op).is_some() => self.emit_compound(b, out),
            Expr::Break(b) => self.emit_break(b, following, out),
            Expr::Continue(c) => self.emit_continue(c, following, out),
            Expr::Return(r) => self.emit_return(r, following, out),
            Expr::Macro(m) => {
                if !m.attrs.is_empty() {
                    self.err(&m.attrs[0], "attributes on kernel statements are not supported");
                }
                self.emit_macro(m.mac, following, out);
            }
            Expr::Block(b) if b.label.is_none() && b.attrs.is_empty() => {
                let blk = self.rewrite_block(b.block);
                out.push(Stmt::Expr(
                    Expr::Block(ExprBlock {
                        attrs: Vec::new(),
                        label: None,
                        block: blk,
                    }),
                    semi.then(Default::default),
                ));
            }
            other => {
                let e2 = self.rw(other);
                out.push(Stmt::Expr(e2, semi.then(Default::default)));
            }
        }
    }


    fn emit_if_stmt(&mut self, i: ExprIf, out: &mut Vec<Stmt>) {
        let whole = Expr::If(i);
        let s = scan_expr(&whole);
        let Expr::If(i) = whole else { unreachable!() };
        if !i.attrs.is_empty() {
            self.err(&i.attrs[0], "attributes on kernel statements are not supported");
        }
        let sp = i.if_token.span;
        let else_expr = i.else_branch.map(|(_, e)| *e);
        self.emit_if_core(*i.cond, i.then_branch, else_expr, false, sp, s, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn emit_if_core(
        &mut self,
        cond: Expr,
        then_b: Block,
        else_b: Option<Expr>,
        coherent: bool,
        sp: Span,
        scan: ExitScan,
        out: &mut Vec<Stmt>,
    ) {
        let cond_sp = cond.span();
        let cond2 = self.rw_cond(cond);
        let then_stmts = self.rewrite_block(then_b).stmts;
        let guard = Ident::new(if coherent { "any" } else { "should_branch" }, sp);

        let mut inner: Vec<Stmt> = Vec::new();
        inner.push(parse_quote_spanned!(cond_sp=> let __c = #cond2;));
        inner.push(parse_quote_spanned!(cond_sp=> let __exec1 = __exec.and_cond(__c);));
        inner.push(parse_quote_spanned!(sp=>
            if __exec1.#guard() {
                let __exec = __exec1;
                #(#then_stmts)*
            }
        ));
        if let Some(eb) = else_b {
            let else_stmts: Vec<Stmt> = match eb {
                Expr::Block(b) => {
                    if let Some(lbl) = &b.label {
                        self.err(lbl, "labeled blocks are not supported in #[kernel]");
                    }
                    self.rewrite_block(b.block).stmts
                }
                nested @ Expr::If(_) => self.rewrite_stmts(vec![Stmt::Expr(nested, None)]),
                other => {
                    self.err(&other, "unsupported `else` form in #[kernel]");
                    Vec::new()
                }
            };
            inner.push(parse_quote_spanned!(cond_sp=> let __exec1 = __exec.and_not_cond(__c);));
            inner.push(parse_quote_spanned!(sp=>
                if __exec1.#guard() {
                    let __exec = __exec1;
                    #(#else_stmts)*
                }
            ));
        }
        out.push(block_stmt(inner));
        self.refresh_after(scan, sp, out);
    }


    fn emit_while(&mut self, w: ExprWhile, out: &mut Vec<Stmt>) {
        if !w.attrs.is_empty() {
            self.err(&w.attrs[0], "attributes on kernel statements are not supported");
        }
        if let Some(lbl) = &w.label {
            self.err(lbl, "labeled loops are not supported in #[kernel] (masks target the innermost loop only)");
        }
        if matches!(&*w.cond, Expr::Let(_)) {
            self.err(&w.cond, "`while let` is not supported in #[kernel]; use a plain condition");
            return; 
        }
        let sp = w.while_token.span;
        self.emit_while_core(*w.cond, w.body, sp, out);
    }

    fn emit_while_core(&mut self, cond: Expr, body: Block, sp: Span, out: &mut Vec<Stmt>) {
        let body_scan = scan_block(&body).merge(scan_expr(&cond));
        let has_co = body_scan.co;
        let depth = self.loops.len();
        let lid = Ident::new(&format!("__loop{depth}"), Span::call_site());
        let iid = has_co.then(|| Ident::new(&format!("__iter{depth}"), Span::call_site()));
        let label = has_co.then(|| Lifetime::new(&format!("'__cont{depth}"), Span::call_site()));
        let brk = has_co.then(|| Lifetime::new(&format!("'__brk{depth}"), Span::call_site()));

        self.loops.push(Frame {
            loop_id: lid.clone(),
            iter_id: iid.clone(),
            cont_label: label.clone(),
            brk_label: brk.clone(),
        });
        let cond_sp = cond.span();
        let cond2 = self.rw_cond(cond);
        let body_stmts = self.rewrite_block(body).stmts;
        self.loops.pop();

        let stmt: Stmt = if let (Some(iid), Some(label), Some(brk)) = (&iid, &label, &brk) {
            parse_quote_spanned!(sp=> {
                let mut #lid = __exec.enter_loop(#cond2);
                #brk: loop {
                    if !#lid.any() {
                        break;
                    }
                    let mut #iid = #lid.iter_mask();
                    #label: {
                        let __exec = #lid.current();
                        #(#body_stmts)*
                    }
                    let __exec = #lid.current();
                    let __c = #cond2;
                    #lid = #lid.and_cond(__c);
                }
            })
        } else {
            parse_quote_spanned!(sp=> {
                let mut #lid = __exec.enter_loop(#cond2);
                loop {
                    if !#lid.any() {
                        break;
                    }
                    let __exec = #lid.current();
                    #(#body_stmts)*
                    let __c = #cond2;
                    #lid = #lid.and_cond(__c);
                }
            })
        };
        let _ = cond_sp;
        out.push(stmt);
        self.refresh_after(
            ExitScan {
                br: false,
                co: false,
                ret: body_scan.ret,
            },
            sp,
            out,
        );
    }


    fn emit_for(&mut self, f: ExprForLoop, out: &mut Vec<Stmt>) {
        if !f.attrs.is_empty() {
            self.err(&f.attrs[0], "attributes on kernel statements are not supported");
        }
        if let Some(lbl) = &f.label {
            self.err(lbl, "labeled loops are not supported in #[kernel] (masks target the innermost loop only)");
        }
        match &*f.pat {
            Pat::Ident(pi) => {
                self.check_ident(&pi.ident);
                if pi.by_ref.is_some() || pi.subpat.is_some() {
                    self.err(&f.pat, "unsupported `for` pattern in #[kernel]");
                }
            }
            Pat::Wild(_) => {}
            other => self.err(
                other,
                "destructuring `for` patterns are not supported in #[kernel]; \
                 bind a single identifier",
            ),
        }
        let sp = f.for_token.span;
        let body_scan = scan_block(&f.body);
        let pat = f.pat;
        let iter = f.expr;
        self.check_tokens_reserved(&iter);

        if !body_scan.any_exit() {
            let body2 = self.rewrite_block(f.body);
            out.push(parse_quote_spanned!(sp=> for #pat in #iter #body2));
            return;
        }

        let depth = self.loops.len();
        let lid = Ident::new(&format!("__loop{depth}"), Span::call_site());
        let iid = body_scan
            .co
            .then(|| Ident::new(&format!("__iter{depth}"), Span::call_site()));
        self.loops.push(Frame {
            loop_id: lid.clone(),
            iter_id: iid.clone(),
            cont_label: None,
            brk_label: None,
        });
        let body_stmts = self.rewrite_block(f.body).stmts;
        self.loops.pop();

        let iter_decl: Option<Stmt> =
            iid.as_ref().map(|iid| parse_quote_spanned!(sp=> let mut #iid = #lid.iter_mask();));
        out.push(parse_quote_spanned!(sp=> {
            let mut #lid = EnterLoopN::<N>::enter_loop_n(__exec);
            for #pat in #iter {
                if !#lid.any() {
                    break;
                }
                #iter_decl
                let __exec = #lid.current();
                #(#body_stmts)*
            }
        }));
        self.refresh_after(
            ExitScan {
                br: false,
                co: false,
                ret: body_scan.ret,
            },
            sp,
            out,
        );
    }

    fn emit_loop(&mut self, l: ExprLoop, out: &mut Vec<Stmt>) {
        if !l.attrs.is_empty() {
            self.err(&l.attrs[0], "attributes on kernel statements are not supported");
        }
        if let Some(lbl) = &l.label {
            self.err(lbl, "labeled loops are not supported in #[kernel] (masks target the innermost loop only)");
        }
        let sp = l.loop_token.span;
        let body_scan = scan_block(&l.body);

        if !body_scan.any_exit() {
            let body2 = self.rewrite_block(l.body);
            out.push(parse_quote_spanned!(sp=> loop #body2));
            return;
        }

        let depth = self.loops.len();
        let lid = Ident::new(&format!("__loop{depth}"), Span::call_site());
        let iid = body_scan
            .co
            .then(|| Ident::new(&format!("__iter{depth}"), Span::call_site()));
        self.loops.push(Frame {
            loop_id: lid.clone(),
            iter_id: iid.clone(),
            cont_label: None,
            brk_label: None,
        });
        let body_stmts = self.rewrite_block(l.body).stmts;
        self.loops.pop();

        let iter_decl: Option<Stmt> =
            iid.as_ref().map(|iid| parse_quote_spanned!(sp=> let mut #iid = #lid.iter_mask();));
        out.push(parse_quote_spanned!(sp=> {
            let mut #lid = EnterLoopN::<N>::enter_loop_n(__exec);
            loop {
                if !#lid.any() {
                    break;
                }
                #iter_decl
                let __exec = #lid.current();
                #(#body_stmts)*
            }
        }));
        self.refresh_after(
            ExitScan {
                br: false,
                co: false,
                ret: body_scan.ret,
            },
            sp,
            out,
        );
    }


    fn emit_break(&mut self, b: syn::ExprBreak, following: bool, out: &mut Vec<Stmt>) {
        if let Some(lbl) = &b.label {
            self.err(lbl, "labeled `break` is not supported in #[kernel] (masks target the innermost loop only)");
            return;
        }
        if let Some(v) = &b.expr {
            self.err(v, "`break` with a value is not supported in #[kernel]");
            return;
        }
        let sp = b.break_token.span;
        let Some(frame) = self.accessible_frame() else {
            let msg = if self.barriers.is_empty() {
                "`break` outside of a loop"
            } else {
                "`break` cannot cross a `foreach!`/`unmasked!` boundary in #[kernel]: the block body is a barrier; it may only target a loop fully inside it"
            };
            self.errors.push(Error::new(sp, msg));
            return;
        };
        let lid = frame.loop_id.clone();
        match frame.brk_label.clone() {
            Some(brk) => out.push(parse_quote_spanned!(sp=>
                if __exec.is_statically_uniform() {
                    break #brk;
                }
            )),
            None => out.push(parse_quote_spanned!(sp=>
                if __exec.is_statically_uniform() {
                    break;
                }
            )),
        }
        out.push(parse_quote_spanned!(sp=> #lid.remove(__exec);));
        if following {
            out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&#lid);));
        }
    }

    fn emit_continue(&mut self, c: syn::ExprContinue, following: bool, out: &mut Vec<Stmt>) {
        if let Some(lbl) = &c.label {
            self.err(lbl, "labeled `continue` is not supported in #[kernel]");
            return;
        }
        let sp = c.continue_token.span;
        let Some(frame) = self.accessible_frame() else {
            let msg = if self.barriers.is_empty() {
                "`continue` outside of a loop"
            } else {
                "`continue` cannot cross a `foreach!`/`unmasked!` boundary in #[kernel]: the block body is a barrier; it may only target a loop fully inside it"
            };
            self.errors.push(Error::new(sp, msg));
            return;
        };
        let Some(iid) = frame.iter_id.clone() else {
            self.errors.push(Error::new(sp, "internal: continue without iteration mask"));
            return;
        };
        match frame.cont_label.clone() {
            Some(label) => out.push(parse_quote_spanned!(sp=>
                if __exec.is_statically_uniform() {
                    break #label;
                }
            )),
            None => out.push(parse_quote_spanned!(sp=>
                if __exec.is_statically_uniform() {
                    continue;
                }
            )),
        }
        out.push(parse_quote_spanned!(sp=> #iid.remove(__exec);));
        if following {
            out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&#iid);));
        }
    }

    fn emit_return(&mut self, r: syn::ExprReturn, following: bool, out: &mut Vec<Stmt>) {
        let sp = r.return_token.span;
        if let Some(b) = self.barriers.last() {
            let what = match b.kind {
                BarrierKind::Unmasked => "`unmasked!`",
                BarrierKind::Foreach => "`foreach!`",
            };
            self.errors.push(Error::new(
                sp,
                format!("`return` inside {what} is not supported in #[kernel] v1"),
            ));
            return;
        }
        match (&self.ret_mode, r.expr) {
            (RetMode::Value(_), Some(v)) => {
                let v2 = self.rw(*v);
                out.push(parse_quote_spanned!(sp=> __ret.masked_assign(__exec, #v2);));
                out.push(parse_quote_spanned!(sp=>
                    if __exec.is_statically_uniform() {
                        return __ret;
                    }
                ));
            }
            (RetMode::Value(_), None) => {
                self.errors.push(Error::new(sp, "this kernel must return a value"));
                return;
            }
            (RetMode::Unit, None) => {
                out.push(parse_quote_spanned!(sp=>
                    if __exec.is_statically_uniform() {
                        return;
                    }
                ));
            }
            (RetMode::Unit, Some(v)) => {
                self.err(&v, "this kernel returns no value");
                return;
            }
        }
        out.push(parse_quote_spanned!(sp=> __fn.remove(__exec);));
        for f in &self.loops {
            let lid = &f.loop_id;
            out.push(parse_quote_spanned!(sp=> #lid.remove(__exec);));
        }
        if following {
            match self.loops.last() {
                Some(f) => {
                    let lid = f.loop_id.clone();
                    out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&#lid);));
                }
                None => out.push(parse_quote_spanned!(sp=> let __exec = __exec.refresh(&__fn);)),
            }
        }
    }


    fn emit_assign(&mut self, a: syn::ExprAssign, out: &mut Vec<Stmt>) {
        let sp = a.eq_token.span;
        let lhs = strip_parens(*a.left);
        let rhs2 = self.rw(*a.right);
        if is_place(&lhs) {
            self.check_place_root(&lhs);
            out.push(parse_quote_spanned!(sp=> (#lhs).masked_assign(__exec, #rhs2);));
        } else if let Expr::Index(ix) = lhs {
            let base = strip_parens(*ix.expr);
            self.check_place_root(&base);
            if !is_place(&base) {
                self.err(
                    &base,
                    "unsupported assignment target in #[kernel]: the indexed value must be a \
                     plain place (a variable, field path, or deref of one)",
                );
                return;
            }
            let idx2 = self.rw(*ix.index);
            out.push(parse_quote_spanned!(sp=> (#base).spmd_write(#idx2, __exec, #rhs2);));
        } else {
            self.err(
                &lhs,
                "unsupported assignment target in #[kernel]: supported forms are `x = v`, \
                 `x.field = v`, `*p = v` and `a[i] = v`; for `a[i].field = v`, \
                 read-modify-write a local instead: `let mut t = a[i]; t.field = v; a[i] = t`",
            );
        }
    }

    fn emit_compound(&mut self, b: ExprBinary, out: &mut Vec<Stmt>) {
        let Some(op) = assign_binop(&b.op) else {
            unreachable!()
        };
        let sp = b.op.span();
        let lhs = strip_parens(*b.left);
        let rhs2 = self.rw(*b.right);
        if is_place(&lhs) {
            self.check_place_root(&lhs);
            out.push(parse_quote_spanned!(sp=>
                (#lhs).masked_assign(__exec, (#lhs) #op (#rhs2));
            ));
        } else if let Expr::Index(ix) = lhs {
            let base = strip_parens(*ix.expr);
            self.check_place_root(&base);
            if !is_place(&base) {
                self.err(
                    &base,
                    "unsupported compound-assignment target in #[kernel]: the indexed value \
                     must be a plain place",
                );
                return;
            }
            let idx2 = self.rw(*ix.index);
            out.push(parse_quote_spanned!(sp=> {
                let __i = #idx2;
                let __t = (#base).spmd_read(__i, __exec) #op (#rhs2);
                (#base).spmd_write(__i, __exec, __t);
            }));
        } else {
            self.err(
                &lhs,
                "unsupported compound-assignment target in #[kernel]: supported forms are \
                 `x op= v`, `x.field op= v`, `*p op= v` and `a[i] op= v`",
            );
        }
    }


    fn emit_macro(&mut self, mac: Macro, _following: bool, out: &mut Vec<Stmt>) {
        let name = mac
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        match name.as_str() {
            "unmasked" => self.emit_unmasked(mac, out),
            "foreach" => self.emit_foreach(mac, out),
            "foreach_2d" => self.emit_foreach_2d(mac, out),
            "foreach_tiled" => self.emit_foreach_tiled(mac, out),
            "cif" => self.emit_cif(mac, out),
            "cwhile" => self.emit_cwhile(mac, out),
            _ => self.err(
                &mac.path,
                &format!("unknown macro inside #[kernel]; {ALLOWED_MACROS}"),
            ),
        }
    }

    fn emit_unmasked(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let stmts = match mac.parse_body_with(Block::parse_within) {
            Ok(s) => s,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        self.barriers.push(Barrier {
            kind: BarrierKind::Unmasked,
            loops_len: self.loops.len(),
        });
        let stmts2 = self.rewrite_stmts(stmts);
        self.barriers.pop();
        out.push(parse_quote_spanned!(sp=> {
            let __exec = ::spmd::AllOn;
            #(#stmts2)*
        }));
    }

    fn emit_foreach(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let input: ForeachInput = match mac.parse_body() {
            Ok(i) => i,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        self.check_ident(&input.var);
        let start2 = self.rw(input.start);
        let end2 = self.rw(input.end);
        self.barriers.push(Barrier {
            kind: BarrierKind::Foreach,
            loops_len: self.loops.len(),
        });
        let body2 = self.rewrite_block(input.body);
        self.barriers.pop();
        let var = input.var;
        out.push(parse_quote_spanned!(sp=> {
            let mut __base: usize = #start2;
            let __n: usize = #end2;
            while __base + N <= __n {
                let #var = ::spmd::LinearIndex::<N>::new(__base);
                #body2
                __base += N;
            }
            if __base < __n {
                let __exec = __exec.and_cond(::spmd::VMask::<N>::first(__n - __base).0);
                let #var = ::spmd::LinearIndex::<N>::new(__base);
                #body2
            }
        }));
    }

    fn emit_foreach_2d(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let input: Foreach2dInput = match mac.parse_body() {
            Ok(i) => i,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        self.check_ident(&input.yvar);
        self.check_ident(&input.xvar);
        let ystart2 = self.rw(input.ystart);
        let yend2 = self.rw(input.yend);
        let xstart2 = self.rw(input.xstart);
        let xend2 = self.rw(input.xend);
        self.barriers.push(Barrier {
            kind: BarrierKind::Foreach,
            loops_len: self.loops.len(),
        });
        let body2 = self.rewrite_block(input.body);
        self.barriers.pop();
        let (yvar, xvar) = (input.yvar, input.xvar);
        out.push(parse_quote_spanned!(sp=> {
            let __y1: usize = #yend2;
            let __x0: usize = #xstart2;
            let __x1: usize = #xend2;
            let mut __y: usize = #ystart2;
            while __y < __y1 {
                let #yvar = __y;
                let mut __base: usize = __x0;
                while __base + N <= __x1 {
                    let #xvar = ::spmd::LinearIndex::<N>::new(__base);
                    #body2
                    __base += N;
                }
                if __base < __x1 {
                    let __exec = __exec.and_cond(::spmd::VMask::<N>::first(__x1 - __base).0);
                    let #xvar = ::spmd::LinearIndex::<N>::new(__base);
                    #body2
                }
                __y += 1;
            }
        }));
    }

    fn emit_foreach_tiled(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let input: Foreach2dInput = match mac.parse_body() {
            Ok(i) => i,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        self.check_ident(&input.yvar);
        self.check_ident(&input.xvar);
        let ystart2 = self.rw(input.ystart);
        let yend2 = self.rw(input.yend);
        let xstart2 = self.rw(input.xstart);
        let xend2 = self.rw(input.xend);
        self.barriers.push(Barrier {
            kind: BarrierKind::Foreach,
            loops_len: self.loops.len(),
        });
        let body2 = self.rewrite_block(input.body);
        self.barriers.pop();
        let (yvar, xvar) = (input.yvar, input.xvar);
        out.push(parse_quote_spanned!(sp=> {
            let __ty: usize = if N >= 16 { 4 } else if N >= 4 { 2 } else { 1 };
            let __tx: usize = N / __ty;
            let __dy: ::spmd::Varying<i32, N> =
                ::spmd::Varying::from_array(::core::array::from_fn(|__l| (__l / __tx) as i32));
            let __dx: ::spmd::Varying<i32, N> =
                ::spmd::Varying::from_array(::core::array::from_fn(|__l| (__l % __tx) as i32));
            let __y1: usize = #yend2;
            let __x0: usize = #xstart2;
            let __x1: usize = #xend2;
            let mut __by: usize = #ystart2;
            while __by < __y1 {
                let mut __bx: usize = __x0;
                while __bx < __x1 {
                    let #yvar = ::spmd::Varying::<i32, N>::splat(__by as i32) + __dy;
                    let #xvar = ::spmd::Varying::<i32, N>::splat(__bx as i32) + __dx;
                    if __by + __ty <= __y1 && __bx + __tx <= __x1 {
                        #body2
                    } else {
                        let __exec = __exec
                            .and_cond(#yvar.spmd_lt(__y1 as i32) & #xvar.spmd_lt(__x1 as i32));
                        #body2
                    }
                    __bx += __tx;
                }
                __by += __ty;
            }
        }));
    }

    fn emit_cif(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let input: CifInput = match mac.parse_body() {
            Ok(i) => i,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        let mut s = scan_expr(&input.cond).merge(scan_block(&input.then));
        if let Some(eb) = &input.else_ {
            s = s.merge(scan_block(eb));
        }
        let else_expr = input.else_.map(|b| {
            Expr::Block(ExprBlock {
                attrs: Vec::new(),
                label: None,
                block: b,
            })
        });
        self.emit_if_core(input.cond, input.then, else_expr, true, sp, s, out);
    }

    fn emit_cwhile(&mut self, mac: Macro, out: &mut Vec<Stmt>) {
        let sp = mac.span();
        let input: CondBlockInput = match mac.parse_body() {
            Ok(i) => i,
            Err(e) => {
                self.errors.push(e);
                return;
            }
        };
        self.emit_while_core(input.cond, input.body, sp, out);
    }


    fn rw(&mut self, e: Expr) -> Expr {
        match e {
            Expr::Array(mut a) => {
                a.elems = a.elems.into_iter().map(|e| self.rw(e)).collect();
                Expr::Array(a)
            }
            Expr::Assign(a) => {
                self.err(&a, "assignment is only supported as a statement in #[kernel]");
                Expr::Assign(a)
            }
            Expr::Async(x) => {
                self.err(&x, "`async` is not supported in #[kernel]");
                Expr::Async(x)
            }
            Expr::Await(x) => {
                self.err(&x, "`.await` is not supported in #[kernel]");
                Expr::Await(x)
            }
            Expr::Binary(b) => self.rw_binary(b),
            Expr::Block(mut b) => {
                if let Some(lbl) = &b.label {
                    self.err(lbl, "labeled blocks are not supported in #[kernel]");
                }
                b.block = self.rewrite_value_block(b.block);
                Expr::Block(b)
            }
            e @ (Expr::Break(_) | Expr::Continue(_) | Expr::Return(_)) => {
                self.err(
                    &e,
                    "`break`/`continue`/`return` are only supported as statements in #[kernel]",
                );
                e
            }
            Expr::Call(c) => self.rw_call(c),
            Expr::Cast(c) => {
                let sp = c.as_token.span;
                let inner = self.rw(*c.expr);
                let ty = c.ty;
                parse_quote_spanned!(sp=> ::spmd::SpmdCast::<#ty>::spmd_cast(#inner))
            }
            Expr::Closure(c) => {
                self.err(
                    &c,
                    "closures are not supported in #[kernel]; use `foreach!` for iteration and \
                     plain kernel calls for abstraction",
                );
                Expr::Closure(c)
            }
            Expr::Const(c) => {
                self.err(&c, "`const` blocks are not supported in #[kernel]");
                Expr::Const(c)
            }
            Expr::Field(mut f) => {
                *f.base = self.rw(*f.base);
                Expr::Field(f)
            }
            e @ (Expr::ForLoop(_) | Expr::Loop(_) | Expr::While(_) | Expr::If(_)) => {
                self.err(
                    &e,
                    "control flow cannot produce a value in #[kernel] v1: use it in statement \
                     position and assign to a variable (or use `Varying::select` for a \
                     branchless pick)",
                );
                e
            }
            Expr::Group(mut g) => {
                *g.expr = self.rw(*g.expr);
                Expr::Group(g)
            }
            Expr::Index(ix) => {
                let sp = ix.bracket_token.span.join();
                let base = self.rw(*ix.expr);
                let idx = self.rw(*ix.index);
                parse_quote_spanned!(sp=> (#base).spmd_read(#idx, __exec))
            }
            Expr::Infer(x) => {
                self.err(&x, "`_` expressions are not supported in #[kernel]");
                Expr::Infer(x)
            }
            Expr::Let(l) => {
                self.err(
                    &l,
                    "`if let`/`while let`/let-chains are not supported in #[kernel]; use plain \
                     conditions",
                );
                Expr::Let(l)
            }
            e @ Expr::Lit(_) => e,
            Expr::Macro(m) => {
                self.err(
                    &m,
                    &format!(
                        "macro calls are not supported in #[kernel] expressions; {ALLOWED_MACROS} \
                         (statement position only)"
                    ),
                );
                Expr::Macro(m)
            }
            Expr::Match(m) => {
                self.err(
                    &m,
                    "`match` is not supported in #[kernel] v1 (including on uniform values); \
                     rewrite as an if/else chain",
                );
                Expr::Match(m)
            }
            Expr::MethodCall(mut m) => {
                *m.receiver = self.rw(*m.receiver);
                m.args = m
                    .args
                    .into_iter()
                    .map(|a| self.rw_call_arg(a, false))
                    .collect();
                Expr::MethodCall(m)
            }
            Expr::Paren(mut p) => {
                *p.expr = self.rw(*p.expr);
                Expr::Paren(p)
            }
            Expr::Path(mut p) => {
                if let Some(first) = p.path.segments.first() {
                    if first.ident.to_string().starts_with("__") {
                        self.err(
                            &p,
                            "identifiers starting with `__` are reserved by #[kernel] machinery",
                        );
                    }
                }
                widen_path(&mut p.path);
                Expr::Path(p)
            }
            Expr::Range(mut r) => {
                r.start = r.start.map(|s| Box::new(self.rw(*s)));
                r.end = r.end.map(|s| Box::new(self.rw(*s)));
                Expr::Range(r)
            }
            Expr::Reference(mut r) => {
                if r.mutability.is_some() {
                    self.err(
                        &r,
                        "`&mut` borrows are not supported in #[kernel] expressions (pass `&mut` \
                         slices only as whole arguments to kernel calls)",
                    );
                }
                *r.expr = self.rw(*r.expr);
                Expr::Reference(r)
            }
            Expr::Repeat(mut r) => {
                *r.expr = self.rw(*r.expr);
                Expr::Repeat(r)
            }
            Expr::Struct(mut s) => {
                for f in &mut s.fields {
                    let e = std::mem::replace(&mut f.expr, Expr::PLACEHOLDER);
                    f.expr = self.rw(e);
                }
                s.rest = s.rest.map(|r| Box::new(self.rw(*r)));
                Expr::Struct(s)
            }
            Expr::Try(t) => {
                self.err(&t, "`?` is not supported in #[kernel]");
                Expr::Try(t)
            }
            Expr::TryBlock(t) => {
                self.err(&t, "`try` blocks are not supported in #[kernel]");
                Expr::TryBlock(t)
            }
            Expr::Tuple(mut t) => {
                t.elems = t.elems.into_iter().map(|e| self.rw(e)).collect();
                Expr::Tuple(t)
            }
            Expr::Unary(mut u) => {
                *u.expr = self.rw(*u.expr);
                Expr::Unary(u)
            }
            Expr::Unsafe(u) => {
                self.err(
                    &u,
                    "`unsafe` blocks are not supported in #[kernel] v1 (unchecked memory access \
                     will arrive with a dedicated opt-in)",
                );
                Expr::Unsafe(u)
            }
            Expr::Yield(y) => {
                self.err(&y, "`yield` is not supported in #[kernel]");
                Expr::Yield(y)
            }
            Expr::Verbatim(v) => {
                if v.is_empty() {
                    Expr::Verbatim(v)
                } else {
                    self.errors
                        .push(Error::new_spanned(&v, "unsupported expression in #[kernel]"));
                    Expr::Verbatim(v)
                }
            }
            other => {
                self.err(&other, "unsupported expression in #[kernel]");
                other
            }
        }
    }

    fn rewrite_value_block(&mut self, b: Block) -> Block {
        let mut stmts = b.stmts;
        let tail: Option<Expr> = match stmts.last() {
            Some(Stmt::Expr(e, None)) if is_tail_value(e) => {
                let Some(Stmt::Expr(e, _)) = stmts.pop() else {
                    unreachable!()
                };
                Some(e)
            }
            _ => None,
        };
        let mut out = self.rewrite_stmts(stmts);
        if let Some(t) = tail {
            let t2 = self.rw(t);
            out.push(Stmt::Expr(t2, None));
        }
        Block {
            brace_token: b.brace_token,
            stmts: out,
        }
    }

    fn rw_binary(&mut self, b: ExprBinary) -> Expr {
        if assign_binop(&b.op).is_some() {
            self.err(
                &b,
                "compound assignment is only supported as a statement in #[kernel]",
            );
            return Expr::Binary(b);
        }
        let sp = b.op.span();
        match cmp_method(&b.op) {
            Some(name) => {
                let m = Ident::new(name, sp);
                let l = self.rw(*b.left);
                let r = self.rw(*b.right);
                let recv = paren_for_receiver(l);
                parse_quote_spanned!(sp=> #recv.#m(#r))
            }
            None => match &b.op {
                BinOp::And(_) => {
                    let l = self.rw(*b.left);
                    let r = self.rw(*b.right);
                    let recv = paren_for_receiver(l);
                    parse_quote_spanned!(sp=> #recv.spmd_and(|| #r))
                }
                BinOp::Or(_) => {
                    let l = self.rw(*b.left);
                    let r = self.rw(*b.right);
                    let recv = paren_for_receiver(l);
                    parse_quote_spanned!(sp=> #recv.spmd_or(|| #r))
                }
                _ => {
                    let mut b = b;
                    *b.left = self.rw(*b.left);
                    *b.right = self.rw(*b.right);
                    Expr::Binary(b)
                }
            },
        }
    }

    fn rw_call(&mut self, c: syn::ExprCall) -> Expr {
        let sp = c.paren_token.span.join();
        match *c.func {
            Expr::Path(mut p) => {
                widen_path(&mut p.path);
                let single = p.qself.is_none()
                    && p.path.leading_colon.is_none()
                    && p.path.segments.len() == 1;
                let is_kernel_call = single && {
                    let seg = &p.path.segments[0];
                    let name = seg.ident.to_string();
                    if name.starts_with("__") {
                        self.err(&p, "identifiers starting with `__` are reserved by #[kernel] machinery");
                    }
                    name.chars()
                        .next()
                        .map(|ch| ch.is_lowercase() || ch == '_')
                        .unwrap_or(false)
                };
                if is_kernel_call {
                    let seg = &p.path.segments[0];
                    if !seg.arguments.is_empty() {
                        self.err(
                            &seg.arguments,
                            "explicit generic arguments on kernel calls are not supported; the \
                             macro supplies `::<N, _>` itself",
                        );
                    }
                    let name = seg.ident.clone();
                    let args: Vec<Expr> = c
                        .args
                        .into_iter()
                        .map(|a| self.rw_call_arg(a, true))
                        .collect();
                    parse_quote_spanned!(sp=> #name::<N, _>(__exec #(, #args)*))
                } else {
                    let args: Vec<Expr> = c
                        .args
                        .into_iter()
                        .map(|a| self.rw_call_arg(a, false))
                        .collect();
                    parse_quote_spanned!(sp=> #p(#(#args),*))
                }
            }
            other => {
                self.err(
                    &other,
                    "indirect calls (through values or closures) are not supported in #[kernel]",
                );
                other
            }
        }
    }

    fn rw_call_arg(&mut self, a: Expr, kernel_call: bool) -> Expr {
        if let Expr::Reference(mut r) = a {
            if r.mutability.is_some() && !kernel_call {
                self.err(
                    &r,
                    "passing `&mut` to a non-kernel function is not supported in #[kernel] \
                     (masked lanes could be observed mid-update); pass `&mut` slices to \
                     #[kernel] fns only",
                );
            }
            *r.expr = self.rw(*r.expr);
            return Expr::Reference(r);
        }
        self.rw(a)
    }


    fn rw_cond(&mut self, e: Expr) -> Expr {
        match e {
            Expr::Unary(u) if matches!(u.op, syn::UnOp::Not(_)) => {
                let sp = u.op.span();
                let inner = self.rw_cond(*u.expr);
                let recv = paren_for_receiver(inner);
                parse_quote_spanned!(sp=> #recv.spmd_not())
            }
            Expr::Paren(mut p) => {
                *p.expr = self.rw_cond(*p.expr);
                Expr::Paren(p)
            }
            Expr::Group(mut g) => {
                *g.expr = self.rw_cond(*g.expr);
                Expr::Group(g)
            }
            Expr::Binary(b) if matches!(b.op, BinOp::And(_) | BinOp::Or(_)) => {
                let sp = b.op.span();
                let method = Ident::new(
                    if matches!(b.op, BinOp::And(_)) {
                        "spmd_and"
                    } else {
                        "spmd_or"
                    },
                    sp,
                );
                let l = self.rw_cond(*b.left);
                let r = self.rw_cond(*b.right);
                let recv = paren_for_receiver(l);
                parse_quote_spanned!(sp=> #recv.#method(|| #r))
            }
            Expr::Let(l) => {
                self.err(
                    &l,
                    "`if let`/`while let` are not supported in #[kernel]; use plain conditions",
                );
                Expr::Let(l)
            }
            other => self.rw(other),
        }
    }
}


fn block_stmt(stmts: Vec<Stmt>) -> Stmt {
    Stmt::Expr(
        Expr::Block(ExprBlock {
            attrs: Vec::new(),
            label: None,
            block: Block {
                brace_token: Default::default(),
                stmts,
            },
        }),
        None,
    )
}

fn strip_parens(e: Expr) -> Expr {
    match e {
        Expr::Paren(p) => strip_parens(*p.expr),
        Expr::Group(g) => strip_parens(*g.expr),
        other => other,
    }
}

fn is_place(e: &Expr) -> bool {
    match e {
        Expr::Path(_) => true,
        Expr::Field(f) => is_place_base(&f.base),
        Expr::Unary(u) => matches!(u.op, syn::UnOp::Deref(_)) && is_place_base(&u.expr),
        _ => false,
    }
}

fn is_place_base(e: &Expr) -> bool {
    match e {
        Expr::Paren(p) => is_place_base(&p.expr),
        Expr::Group(g) => is_place_base(&g.expr),
        other => is_place(other),
    }
}

fn place_root_ident(e: &Expr) -> Option<&Ident> {
    match e {
        Expr::Path(p) => p.path.segments.first().map(|s| &s.ident),
        Expr::Field(f) => place_root_ident_base(&f.base),
        Expr::Unary(u) if matches!(u.op, syn::UnOp::Deref(_)) => place_root_ident_base(&u.expr),
        _ => None,
    }
}

fn place_root_ident_base(e: &Expr) -> Option<&Ident> {
    match e {
        Expr::Paren(p) => place_root_ident_base(&p.expr),
        Expr::Group(g) => place_root_ident_base(&g.expr),
        other => place_root_ident(other),
    }
}

fn cmp_method(op: &BinOp) -> Option<&'static str> {
    Some(match op {
        BinOp::Lt(_) => "spmd_lt",
        BinOp::Le(_) => "spmd_le",
        BinOp::Gt(_) => "spmd_gt",
        BinOp::Ge(_) => "spmd_ge",
        BinOp::Eq(_) => "spmd_eq",
        BinOp::Ne(_) => "spmd_ne",
        _ => return None,
    })
}

fn assign_binop(op: &BinOp) -> Option<BinOp> {
    Some(match op {
        BinOp::AddAssign(t) => BinOp::Add(syn::token::Plus(t.spans[0])),
        BinOp::SubAssign(t) => BinOp::Sub(syn::token::Minus(t.spans[0])),
        BinOp::MulAssign(t) => BinOp::Mul(syn::token::Star(t.spans[0])),
        BinOp::DivAssign(t) => BinOp::Div(syn::token::Slash(t.spans[0])),
        BinOp::RemAssign(t) => BinOp::Rem(syn::token::Percent(t.spans[0])),
        BinOp::BitAndAssign(t) => BinOp::BitAnd(syn::token::And(t.spans[0])),
        BinOp::BitOrAssign(t) => BinOp::BitOr(syn::token::Or(t.spans[0])),
        BinOp::BitXorAssign(t) => BinOp::BitXor(syn::token::Caret(t.spans[0])),
        BinOp::ShlAssign(t) => BinOp::Shl(syn::token::Shl(t.spans[0])),
        BinOp::ShrAssign(t) => BinOp::Shr(syn::token::Shr(t.spans[0])),
        _ => return None,
    })
}

fn paren_for_receiver(e: Expr) -> Expr {
    let safe = matches!(
        e,
        Expr::Path(_)
            | Expr::Lit(_)
            | Expr::Paren(_)
            | Expr::Call(_)
            | Expr::MethodCall(_)
            | Expr::Field(_)
            | Expr::Index(_)
            | Expr::Tuple(_)
            | Expr::Array(_)
    );
    if safe {
        e
    } else {
        Expr::Paren(syn::ExprParen {
            attrs: Vec::new(),
            paren_token: Default::default(),
            expr: Box::new(e),
        })
    }
}


pub struct ForeachInput {
    pub var: Ident,
    pub start: Expr,
    pub end: Expr,
    pub body: Block,
}

fn parse_axis(input: ParseStream) -> syn::Result<(Ident, Expr, Expr)> {
    let var: Ident = input.parse()?;
    input.parse::<Token![in]>()?;
    let range = Expr::parse_without_eager_brace(input)?;
    let Expr::Range(r) = range else {
        return Err(Error::new_spanned(
            &range,
            "expected a `start..end` range (e.g. `i in 0..n`)",
        ));
    };
    if !matches!(r.limits, RangeLimits::HalfOpen(_)) {
        return Err(Error::new_spanned(
            &r,
            "inclusive ranges (`..=`) are not supported; use `start..end`",
        ));
    }
    let (Some(start), Some(end)) = (r.start.clone(), r.end.clone()) else {
        return Err(Error::new_spanned(
            &r,
            "both range endpoints are required (e.g. `0..n`)",
        ));
    };
    Ok((var, *start, *end))
}

impl Parse for ForeachInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (var, start, end) = parse_axis(input)?;
        let body: Block = input.parse()?;
        Ok(ForeachInput {
            var,
            start,
            end,
            body,
        })
    }
}

pub struct Foreach2dInput {
    pub yvar: Ident,
    pub ystart: Expr,
    pub yend: Expr,
    pub xvar: Ident,
    pub xstart: Expr,
    pub xend: Expr,
    pub body: Block,
}

impl Parse for Foreach2dInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let (yvar, ystart, yend) = parse_axis(input)?;
        input.parse::<Token![,]>()?;
        let (xvar, xstart, xend) = parse_axis(input)?;
        let body: Block = input.parse()?;
        Ok(Foreach2dInput {
            yvar,
            ystart,
            yend,
            xvar,
            xstart,
            xend,
            body,
        })
    }
}

pub struct CifInput {
    pub cond: Expr,
    pub then: Block,
    pub else_: Option<Block>,
}

impl Parse for CifInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let cond: Expr = Expr::parse_without_eager_brace(input)?;
        input.parse::<Token![=>]>()?;
        let then: Block = input.parse()?;
        let else_ = if input.peek(Token![else]) {
            input.parse::<Token![else]>()?;
            Some(input.parse::<Block>()?)
        } else {
            None
        };
        Ok(CifInput { cond, then, else_ })
    }
}

pub struct CondBlockInput {
    pub cond: Expr,
    pub body: Block,
}

impl Parse for CondBlockInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let cond: Expr = Expr::parse_without_eager_brace(input)?;
        input.parse::<Token![=>]>()?;
        let body: Block = input.parse()?;
        Ok(CondBlockInput { cond, body })
    }
}
