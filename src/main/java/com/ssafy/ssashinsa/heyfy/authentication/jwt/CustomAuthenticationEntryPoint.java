package com.ssafy.ssashinsa.heyfy.authentication.jwt;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.ssafy.ssashinsa.heyfy.common.ErrorCode;
import com.ssafy.ssashinsa.heyfy.common.ErrorResponse;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.AuthenticationException;
import org.springframework.security.web.AuthenticationEntryPoint;
import org.springframework.stereotype.Component;

import java.io.IOException;



@Component
public class CustomAuthenticationEntryPoint implements AuthenticationEntryPoint {

    @Override
    public void commence(HttpServletRequest request, HttpServletResponse response, AuthenticationException authException) throws IOException, ServletException {
        // ErrorResponse에서 제공하는 정적 팩토리 메서드를 사용하여 응답 생성
        ResponseEntity<ErrorResponse> responseEntity = ErrorResponse.responseEntity(ErrorCode.UNAUTHORIZED);

        // 응답 상태와 컨텐츠 타입 설정
        response.setStatus(responseEntity.getStatusCode().value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);

        // ObjectMapper를 사용해 응답 본문을 JSON으로 변환하여 작성
        ObjectMapper objectMapper = new ObjectMapper();
        objectMapper.writeValue(response.getOutputStream(), responseEntity.getBody());
    }
}