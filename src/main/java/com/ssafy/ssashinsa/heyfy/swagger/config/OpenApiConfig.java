package com.ssafy.ssashinsa.heyfy.swagger.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.examples.Example;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.servers.Server;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.util.StringUtils;

import java.util.List;

@Configuration
public class OpenApiConfig {
    final String securitySchemeName = "bearerAuth";
    final String apiTitle = String.format("%s API", StringUtils.capitalize("HeyFY"));
    Info info = new Info()
            .version("v0.1.0")
            .title("HeyFY")
            .description("토큰 입력 시 Bearer 꼭 추가!!");
    @Bean
    public OpenAPI openAPI() {
        Components components = new Components()
                .addExamples("MissingRequired", new Example()
                        .summary("필수 정보 누락")
                        .value("{\"status\":400," +
                                "\"httpError\":\"BAD_REQUEST\"," +
                                "\"errorCode\":\"INVALID_FIELD\"," +
                                "\"message\":\"must not be blank\"}"))
                .addExamples("NotFound", new Example()
                        .summary("리소스 없음")
                        .value("{\"status\":404," +
                                "\"httpError\":\"NOT_FOUND\"," +
                                "\"errorCode\":\"NOT_FOUND\"," +
                                "\"message\":\"리소스를 찾을 수 없습니다\"}"))
                .addExamples("Conflict", new Example()
                        .summary("충돌")
                        .value("{\"code\":409,\"message\":\"Conflict\"}"))
                .addExamples("InternalError", new Example()
                        .summary("서버 에러")
                        .value("{\"status\":500," +
                                "\"httpError\":\"INTERNAL_SERVER_ERROR\"," +
                                "\"errorCode\":\"INTERNAL_SERVER_ERROR\"," +
                                "\"message\":\"에러가 발생했습니다\"}"));

        return new OpenAPI()
                .servers(List.of(new Server().url("/").description(apiTitle)))
                .components(components.addSecuritySchemes(securitySchemeName,
                        new SecurityScheme()
                                .name(securitySchemeName)
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("JWT")
                ))
                .info(info);
    }
}