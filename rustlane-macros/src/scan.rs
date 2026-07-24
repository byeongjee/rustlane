use syn::{Block, Expr, Macro, Stmt};

use crate::rewrite::{CifInput, CondBlockInput};

#[derive(Debug, Default, Clone, Copy)]
pub struct ExitScan {
    pub br: bool,
    pub co: bool,
    pub ret: bool,
}

impl ExitScan {
    pub fn merge(self, o: ExitScan) -> ExitScan {
        ExitScan {
            br: self.br || o.br,
            co: self.co || o.co,
            ret: self.ret || o.ret,
        }
    }

    pub fn loop_mask(self) -> ExitScan {
        ExitScan {
            br: false,
            co: false,
            ret: self.ret,
        }
    }

    pub fn any_exit(self) -> bool {
        self.br || self.co || self.ret
    }
}

pub fn scan_block(b: &Block) -> ExitScan {
    b.stmts
        .iter()
        .map(scan_stmt)
        .fold(ExitScan::default(), ExitScan::merge)
}

fn scan_stmt(s: &Stmt) -> ExitScan {
    match s {
        Stmt::Local(l) => l
            .init
            .as_ref()
            .map(|i| scan_expr(&i.expr))
            .unwrap_or_default(),
        Stmt::Item(_) => ExitScan::default(),
        Stmt::Expr(e, _) => scan_expr(e),
        Stmt::Macro(m) => scan_macro(&m.mac),
    }
}

pub fn scan_expr(e: &Expr) -> ExitScan {
    match e {
        Expr::Break(_) => ExitScan {
            br: true,
            ..Default::default()
        },
        Expr::Continue(_) => ExitScan {
            co: true,
            ..Default::default()
        },
        Expr::Return(_) => ExitScan {
            ret: true,
            ..Default::default()
        },
        Expr::ForLoop(f) => scan_block(&f.body).loop_mask(),
        Expr::While(w) => scan_expr(&w.cond).merge(scan_block(&w.body)).loop_mask(),
        Expr::Loop(l) => scan_block(&l.body).loop_mask(),
        Expr::If(i) => {
            let mut s = scan_expr(&i.cond).merge(scan_block(&i.then_branch));
            if let Some((_, eb)) = &i.else_branch {
                s = s.merge(scan_expr(eb));
            }
            s
        }
        Expr::Binary(b) => scan_expr(&b.left).merge(scan_expr(&b.right)),
        Expr::Assign(a) => scan_expr(&a.left).merge(scan_expr(&a.right)),
        Expr::Unary(u) => scan_expr(&u.expr),
        Expr::Paren(p) => scan_expr(&p.expr),
        Expr::Group(g) => scan_expr(&g.expr),
        Expr::Block(b) => scan_block(&b.block),
        Expr::Unsafe(u) => scan_block(&u.block),
        Expr::Call(c) => c
            .args
            .iter()
            .map(scan_expr)
            .fold(scan_expr(&c.func), ExitScan::merge),
        Expr::MethodCall(m) => m
            .args
            .iter()
            .map(scan_expr)
            .fold(scan_expr(&m.receiver), ExitScan::merge),
        Expr::Index(i) => scan_expr(&i.expr).merge(scan_expr(&i.index)),
        Expr::Field(f) => scan_expr(&f.base),
        Expr::Cast(c) => scan_expr(&c.expr),
        Expr::Reference(r) => scan_expr(&r.expr),
        Expr::Array(a) => a
            .elems
            .iter()
            .map(scan_expr)
            .fold(ExitScan::default(), ExitScan::merge),
        Expr::Tuple(t) => t
            .elems
            .iter()
            .map(scan_expr)
            .fold(ExitScan::default(), ExitScan::merge),
        Expr::Repeat(r) => scan_expr(&r.expr),
        Expr::Struct(s) => {
            let mut acc = s
                .fields
                .iter()
                .map(|f| scan_expr(&f.expr))
                .fold(ExitScan::default(), ExitScan::merge);
            if let Some(rest) = &s.rest {
                acc = acc.merge(scan_expr(rest));
            }
            acc
        }
        Expr::Range(r) => {
            let mut acc = ExitScan::default();
            if let Some(s) = &r.start {
                acc = acc.merge(scan_expr(s));
            }
            if let Some(e) = &r.end {
                acc = acc.merge(scan_expr(e));
            }
            acc
        }
        Expr::Let(l) => scan_expr(&l.expr),
        Expr::Match(m) => m
            .arms
            .iter()
            .map(|a| scan_expr(&a.body))
            .fold(scan_expr(&m.expr), ExitScan::merge),
        Expr::Macro(m) => scan_macro(&m.mac),
        _ => ExitScan::default(),
    }
}

pub fn scan_macro(mac: &Macro) -> ExitScan {
    let name = mac
        .path
        .segments
        .last()
        .map(|s| s.ident.to_string())
        .unwrap_or_default();
    match name.as_str() {
        "cif" => match syn::parse2::<CifInput>(mac.tokens.clone()) {
            Ok(c) => {
                let mut s = scan_expr(&c.cond).merge(scan_block(&c.then));
                if let Some(eb) = &c.else_ {
                    s = s.merge(scan_block(eb));
                }
                s
            }
            Err(_) => ExitScan::default(),
        },
        "cwhile" => match syn::parse2::<CondBlockInput>(mac.tokens.clone()) {
            Ok(c) => scan_expr(&c.cond).merge(scan_block(&c.body)).loop_mask(),
            Err(_) => ExitScan::default(),
        },
        _ => ExitScan::default(),
    }
}
