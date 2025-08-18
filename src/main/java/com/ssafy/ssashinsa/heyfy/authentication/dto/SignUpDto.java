package com.ssafy.ssashinsa.heyfy.authentication.dto;

import io.swagger.v3.oas.annotations.media.Schema;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Pattern;
import lombok.Data;

@Data
@Schema(description = "회원가입 요청 DTO")
public class SignUpDto {
    @NotBlank
    private String studentId;
    @NotBlank
    @Pattern(regexp = "^(?=.*[a-zA-Z])(?=.*[0-9])(?=.*[!@#$%^&*]).{8,20}$",
            message = "비밀번호는 영문, 숫자, 특수문자(!@#$%^&*)를 포함하여 8~20자로 구성되어야 합니다.")
    private String password;
    @NotBlank
    private String name;
    private String email;
    private String language;
    private String univName;
}
