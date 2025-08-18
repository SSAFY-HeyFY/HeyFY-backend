package com.ssafy.ssashinsa.heyfy.authentication.config;

import java.util.List;
public class PermitAllPaths {
    // 인증 필요 없는 url 패턴 리스트  :  application.yml 등에 데이터 저장하는 방법도 있으나, 관리가 힘들어지므로 현재 방식 저장. 후에 상의하여 결정 예정
    public static final List<String> PATHS = List.of(
            "/auth/signup/**",
            "/auth/signin/**",
            "/auth/refresh/**",
            "/auth/token/access",
            "/api/test/public",
            "/api/transfers",
            "/css/**",
            "/js/**",
            "/images/**",
            "/webjars/**",
            "/swagger-ui/**",
            "/api-docs/**"
    );
}
